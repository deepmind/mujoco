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

#define NAMESPACE ImPlot
#include "dear_imgui_macros.h"
#include <implot.h>
#include <pybind11/eval.h>
#include <pybind11/pybind11.h>
#include <pybind11/pytypes.h>
#include <pybind11/stl.h>

// NOLINTBEGIN(whitespace/line_length)

namespace py = pybind11;
using ImString = const char*;
static constexpr const ImVec2 ImVec2_Zero = ImVec2(0.0f, 0.0f);
static constexpr const ImVec2 ImVec2_One = ImVec2(1.0f, 1.0f);
static constexpr const ImVec2 ImVec2_NegOne_Zero = ImVec2(-1.0f, 0.0f);
static constexpr const ImVec4 ImVec4_One = ImVec4(1.0f, 1.0f, 1.0f, 1.0f);

PYBIND11_MODULE(implot, m) {
  // Import dear_imgui to make types like ImVec2 available.
  py::module_::import("mujoco.studio.dear_imgui.dear_imgui");

  // Each pybind module holds its own ImGui/ImPlot context globals
  m.def(
      "set_imgui_context",
      [](intptr_t ptr) {
        ImGui::SetCurrentContext(reinterpret_cast<ImGuiContext*>(ptr));
      },
      py::arg("ptr"), "Set ImGui context pointer.");
  m.def(
      "set_implot_context",
      [](intptr_t ptr) {
        ImPlot::SetCurrentContext(reinterpret_cast<ImPlotContext*>(ptr));
      },
      py::arg("ptr"), "Set ImPlot context pointer.");

  // Types.
  py::class_<ImPlotPoint>(m, "Point")
      .def(py::init<>())
      .def(py::init<double, double>(), py::arg("_x"), py::arg("_y"))
      .def_readwrite("x", &ImPlotPoint::x)
      .def_readwrite("y", &ImPlotPoint::y);

  py::class_<ImPlotRange>(m, "Range")
      .def(py::init<>())
      .def(py::init<double, double>(), py::arg("_min"), py::arg("_max"))
      .def_readwrite("min", &ImPlotRange::Min)
      .def_readwrite("max", &ImPlotRange::Max);

//   py::class_<ImPlotRect>(m, "Rect")
//       .def(py::init<>())
//       .def(py::init<ImPlotRange, ImPlotRange>(), py::arg("_x"), py::arg("_y"))
//       .def_readwrite("x", &ImPlotRect::X)
//       .def_readwrite("y", &ImPlotRect::Y);

  // Enumerations.

  py::enum_<ImAxis_>(m, "Axis")
      .value("X1", ImAxis_X1)
      .value("X2", ImAxis_X2)
      .value("X3", ImAxis_X3)
      .value("Y1", ImAxis_Y1)
      .value("Y2", ImAxis_Y2)
      .value("Y3", ImAxis_Y3);

  py::enum_<ImPlotFlags_>(m, "Flags")
      .value("None", ImPlotFlags_None)
      .value("NoTitle", ImPlotFlags_NoTitle)
      .value("NoLegend", ImPlotFlags_NoLegend)
      .value("NoMouseText", ImPlotFlags_NoMouseText)
      .value("NoInputs", ImPlotFlags_NoInputs)
      .value("NoMenus", ImPlotFlags_NoMenus)
      .value("NoBoxSelect", ImPlotFlags_NoBoxSelect)
      .value("NoFrame", ImPlotFlags_NoFrame)
      .value("Equal", ImPlotFlags_Equal)
      .value("Crosshairs", ImPlotFlags_Crosshairs)
      .value("CanvasOnly", ImPlotFlags_CanvasOnly);

  py::enum_<ImPlotAxisFlags_>(m, "AxisFlags")
      .value("None", ImPlotAxisFlags_None)
      .value("NoLabel", ImPlotAxisFlags_NoLabel)
      .value("NoGridLines", ImPlotAxisFlags_NoGridLines)
      .value("NoTickMarks", ImPlotAxisFlags_NoTickMarks)
      .value("NoTickLabels", ImPlotAxisFlags_NoTickLabels)
      .value("NoInitialFit", ImPlotAxisFlags_NoInitialFit)
      .value("NoMenus", ImPlotAxisFlags_NoMenus)
      .value("NoSideSwitch", ImPlotAxisFlags_NoSideSwitch)
      .value("NoHighlight", ImPlotAxisFlags_NoHighlight)
      .value("Opposite", ImPlotAxisFlags_Opposite)
      .value("Foreground", ImPlotAxisFlags_Foreground)
      .value("Invert", ImPlotAxisFlags_Invert)
      .value("AutoFit", ImPlotAxisFlags_AutoFit)
      .value("RangeFit", ImPlotAxisFlags_RangeFit)
      .value("PanStretch", ImPlotAxisFlags_PanStretch)
      .value("LockMin", ImPlotAxisFlags_LockMin)
      .value("LockMax", ImPlotAxisFlags_LockMax)
      .value("Lock", ImPlotAxisFlags_Lock)
      .value("NoDecorations", ImPlotAxisFlags_NoDecorations)
      .value("AuxDefault", ImPlotAxisFlags_AuxDefault);

  py::enum_<ImPlotSubplotFlags_>(m, "SubplotFlags")
      .value("None", ImPlotSubplotFlags_None)
      .value("NoTitle", ImPlotSubplotFlags_NoTitle)
      .value("NoLegend", ImPlotSubplotFlags_NoLegend)
      .value("NoMenus", ImPlotSubplotFlags_NoMenus)
      .value("NoResize", ImPlotSubplotFlags_NoResize)
      .value("NoAlign", ImPlotSubplotFlags_NoAlign)
      .value("ShareItems", ImPlotSubplotFlags_ShareItems)
      .value("LinkRows", ImPlotSubplotFlags_LinkRows)
      .value("LinkCols", ImPlotSubplotFlags_LinkCols)
      .value("LinkAllX", ImPlotSubplotFlags_LinkAllX)
      .value("LinkAllY", ImPlotSubplotFlags_LinkAllY)
      .value("ColMajor", ImPlotSubplotFlags_ColMajor);

  py::enum_<ImPlotLegendFlags_>(m, "LegendFlags")
      .value("None", ImPlotLegendFlags_None)
      .value("NoButtons", ImPlotLegendFlags_NoButtons)
      .value("NoHighlightItem", ImPlotLegendFlags_NoHighlightItem)
      .value("NoHighlightAxis", ImPlotLegendFlags_NoHighlightAxis)
      .value("NoMenus", ImPlotLegendFlags_NoMenus)
      .value("Outside", ImPlotLegendFlags_Outside)
      .value("Horizontal", ImPlotLegendFlags_Horizontal)
      .value("Sort", ImPlotLegendFlags_Sort)
      .value("Reverse", ImPlotLegendFlags_Reverse);

  py::enum_<ImPlotMouseTextFlags_>(m, "MouseTextFlags")
      .value("None", ImPlotMouseTextFlags_None)
      .value("NoAuxAxes", ImPlotMouseTextFlags_NoAuxAxes)
      .value("NoFormat", ImPlotMouseTextFlags_NoFormat)
      .value("ShowAlways", ImPlotMouseTextFlags_ShowAlways);

  py::enum_<ImPlotDragToolFlags_>(m, "DragToolFlags")
      .value("None", ImPlotDragToolFlags_None)
      .value("NoCursors", ImPlotDragToolFlags_NoCursors)
      .value("NoFit", ImPlotDragToolFlags_NoFit)
      .value("NoInputs", ImPlotDragToolFlags_NoInputs)
      .value("Delayed", ImPlotDragToolFlags_Delayed);

  py::enum_<ImPlotColormapScaleFlags_>(m, "ColormapScaleFlags")
      .value("None", ImPlotColormapScaleFlags_None)
      .value("NoLabel", ImPlotColormapScaleFlags_NoLabel)
      .value("Opposite", ImPlotColormapScaleFlags_Opposite)
      .value("Invert", ImPlotColormapScaleFlags_Invert);

  py::enum_<ImPlotItemFlags_>(m, "ItemFlags")
      .value("None", ImPlotItemFlags_None)
      .value("NoLegend", ImPlotItemFlags_NoLegend)
      .value("NoFit", ImPlotItemFlags_NoFit);

  py::enum_<ImPlotLineFlags_>(m, "LineFlags")
      .value("None", ImPlotLineFlags_None)
      .value("Segments", ImPlotLineFlags_Segments)
      .value("Loop", ImPlotLineFlags_Loop)
      .value("SkipNaN", ImPlotLineFlags_SkipNaN)
      .value("NoClip", ImPlotLineFlags_NoClip)
      .value("Shaded", ImPlotLineFlags_Shaded);

  py::enum_<ImPlotScatterFlags_>(m, "ScatterFlags")
      .value("None", ImPlotScatterFlags_None)
      .value("NoClip", ImPlotScatterFlags_NoClip);

  py::enum_<ImPlotStairsFlags_>(m, "StairsFlags")
      .value("None", ImPlotStairsFlags_None)
      .value("PreStep", ImPlotStairsFlags_PreStep)
      .value("Shaded", ImPlotStairsFlags_Shaded);

  py::enum_<ImPlotShadedFlags_>(m, "ShadedFlags")
      .value("None", ImPlotShadedFlags_None);

  py::enum_<ImPlotBarsFlags_>(m, "BarsFlags")
      .value("None", ImPlotBarsFlags_None)
      .value("Horizontal", ImPlotBarsFlags_Horizontal);

  py::enum_<ImPlotBarGroupsFlags_>(m, "BarGroupsFlags")
      .value("None", ImPlotBarGroupsFlags_None)
      .value("Horizontal", ImPlotBarGroupsFlags_Horizontal)
      .value("Stacked", ImPlotBarGroupsFlags_Stacked);

  py::enum_<ImPlotErrorBarsFlags_>(m, "ErrorBarsFlags")
      .value("None", ImPlotErrorBarsFlags_None)
      .value("Horizontal", ImPlotErrorBarsFlags_Horizontal);

  py::enum_<ImPlotStemsFlags_>(m, "StemsFlags")
      .value("None", ImPlotStemsFlags_None)
      .value("Horizontal", ImPlotStemsFlags_Horizontal);

  py::enum_<ImPlotInfLinesFlags_>(m, "InfLinesFlags")
      .value("None", ImPlotInfLinesFlags_None)
      .value("Horizontal", ImPlotInfLinesFlags_Horizontal);

  py::enum_<ImPlotPieChartFlags_>(m, "PieChartFlags")
      .value("None", ImPlotPieChartFlags_None)
      .value("Normalize", ImPlotPieChartFlags_Normalize)
      .value("IgnoreHidden", ImPlotPieChartFlags_IgnoreHidden)
      .value("Exploding", ImPlotPieChartFlags_Exploding);

  py::enum_<ImPlotHeatmapFlags_>(m, "HeatmapFlags")
      .value("None", ImPlotHeatmapFlags_None)
      .value("ColMajor", ImPlotHeatmapFlags_ColMajor);

  py::enum_<ImPlotHistogramFlags_>(m, "HistogramFlags")
      .value("None", ImPlotHistogramFlags_None)
      .value("Horizontal", ImPlotHistogramFlags_Horizontal)
      .value("Cumulative", ImPlotHistogramFlags_Cumulative)
      .value("Density", ImPlotHistogramFlags_Density)
      .value("NoOutliers", ImPlotHistogramFlags_NoOutliers)
      .value("ColMajor", ImPlotHistogramFlags_ColMajor);

  py::enum_<ImPlotDigitalFlags_>(m, "DigitalFlags")
      .value("ImPlotNone", ImPlotDigitalFlags_None);

  py::enum_<ImPlotImageFlags_>(m, "ImageFlags")
      .value("None", ImPlotImageFlags_None);

  py::enum_<ImPlotTextFlags_>(m, "TextFlags")
      .value("None", ImPlotTextFlags_None)
      .value("Vertical", ImPlotTextFlags_Vertical);

  py::enum_<ImPlotDummyFlags_>(m, "DummyFlags")
      .value("None", ImPlotDummyFlags_None);

  py::enum_<ImPlotCond_>(m, "Cond")
      .value("None", ImPlotCond_None)
      .value("Always", ImPlotCond_Always)
      .value("Once", ImPlotCond_Once);

  py::enum_<ImPlotCol_>(m, "Col")
      .value("FrameBg", ImPlotCol_FrameBg)
      .value("PlotBg", ImPlotCol_PlotBg)
      .value("PlotBorder", ImPlotCol_PlotBorder)
      .value("LegendBg", ImPlotCol_LegendBg)
      .value("LegendBorder", ImPlotCol_LegendBorder)
      .value("LegendText", ImPlotCol_LegendText)
      .value("TitleText", ImPlotCol_TitleText)
      .value("InlayText", ImPlotCol_InlayText)
      .value("AxisText", ImPlotCol_AxisText)
      .value("AxisGrid", ImPlotCol_AxisGrid)
      .value("AxisTick", ImPlotCol_AxisTick)
      .value("AxisBg", ImPlotCol_AxisBg)
      .value("AxisBgHovered", ImPlotCol_AxisBgHovered)
      .value("AxisBgActive", ImPlotCol_AxisBgActive)
      .value("Selection", ImPlotCol_Selection)
      .value("Crosshairs", ImPlotCol_Crosshairs);

  py::enum_<ImPlotStyleVar_>(m, "StyleVar")
      .value("DigitalSpacing", ImPlotStyleVar_DigitalSpacing)
      .value("PlotBorderSize", ImPlotStyleVar_PlotBorderSize)
      .value("MinorAlpha", ImPlotStyleVar_MinorAlpha)
      .value("MajorTickLen", ImPlotStyleVar_MajorTickLen)
      .value("MinorTickLen", ImPlotStyleVar_MinorTickLen)
      .value("MajorTickSize", ImPlotStyleVar_MajorTickSize)
      .value("MinorTickSize", ImPlotStyleVar_MinorTickSize)
      .value("MajorGridSize", ImPlotStyleVar_MajorGridSize)
      .value("MinorGridSize", ImPlotStyleVar_MinorGridSize)
      .value("PlotPadding", ImPlotStyleVar_PlotPadding)
      .value("LabelPadding", ImPlotStyleVar_LabelPadding)
      .value("LegendPadding", ImPlotStyleVar_LegendPadding)
      .value("LegendInnerPadding", ImPlotStyleVar_LegendInnerPadding)
      .value("LegendSpacing", ImPlotStyleVar_LegendSpacing)
      .value("MousePosPadding", ImPlotStyleVar_MousePosPadding)
      .value("AnnotationPadding", ImPlotStyleVar_AnnotationPadding)
      .value("FitPadding", ImPlotStyleVar_FitPadding)
      .value("PlotDefaultSize", ImPlotStyleVar_PlotDefaultSize)
      .value("PlotMinSize", ImPlotStyleVar_PlotMinSize)
      .value("DigitalPadding", ImPlotStyleVar_DigitalPadding);

  py::enum_<ImPlotScale_>(m, "Scale")
      .value("ImPlotScale_Linear", ImPlotScale_Linear)
      .value("ImPlotScale_Time", ImPlotScale_Time)
      .value("ImPlotScale_Log10", ImPlotScale_Log10)
      .value("ImPlotScale_SymLog", ImPlotScale_SymLog);

  py::enum_<ImPlotMarker_>(m, "Marker")
      .value("None", ImPlotMarker_None)
      .value("Circle", ImPlotMarker_Circle)
      .value("Square", ImPlotMarker_Square)
      .value("Diamond", ImPlotMarker_Diamond)
      .value("Up", ImPlotMarker_Up)
      .value("Down", ImPlotMarker_Down)
      .value("Left", ImPlotMarker_Left)
      .value("Right", ImPlotMarker_Right)
      .value("Cross", ImPlotMarker_Cross)
      .value("Plus", ImPlotMarker_Plus)
      .value("Asterisk", ImPlotMarker_Asterisk);

  py::enum_<ImPlotLocation_>(m, "Location")
      .value("Center", ImPlotLocation_Center)
      .value("North", ImPlotLocation_North)
      .value("South", ImPlotLocation_South)
      .value("West", ImPlotLocation_West)
      .value("East", ImPlotLocation_East)
      .value("NorthWest", ImPlotLocation_NorthWest)
      .value("NorthEast", ImPlotLocation_NorthEast)
      .value("SouthWest", ImPlotLocation_SouthWest)
      .value("SouthEast", ImPlotLocation_SouthEast);

  // Functions.

  DEF3(BeginPlot, (ImString, title_id, ), (const ImVec2&, size, = ImVec2_NegOne_Zero), (ImPlotFlags, flags, = 0));
  DEF0(EndPlot);
  DEF7(BeginSubplots, (ImString, title_id, ), (int, rows, ), (int, cols, ), (const ImVec2&, size, ), (ImPlotSubplotFlags, flags, = 0), (float*, row_ratios, = nullptr), (float*, col_ratios, = nullptr));
  DEF0(EndSubplots);
  DEF3(SetupAxis, (ImAxis, axis, ), (ImString, label, = nullptr), (ImPlotAxisFlags, flags, = 0));
  DEF4(SetupAxisLimits, (ImAxis, axis, ), (double, v_min, ), (double, v_max, ), (ImPlotCond, cond, = ImPlotCond_Once));
  DEF4_F(SetupAxisTicks, (ImAxis, axis, ), (std::vector<double>, values, ), (std::vector<std::string>, labels, ), (bool, keep_default, = false), {
    std::vector<const char*> c_labels;
    c_labels.reserve(labels.size());
    for (const auto& l : labels) {
      c_labels.push_back(l.c_str());
    }
    ImPlot::SetupAxisTicks(axis, values.data(), values.size(), c_labels.empty() ? nullptr : c_labels.data(), keep_default);
  });
  DEF3(SetupAxisLinks, (ImAxis, axis, ), (double*, link_min, ), (double*, link_max, ));
  DEF2(SetupAxisFormat, (ImAxis, axis, ), (ImString, fmt, ));
  DEF2(SetupAxisScale, (ImAxis, axis, ), (ImPlotScale, scale, ));
  DEF3(SetupAxisLimitsConstraints, (ImAxis, axis, ), (double, v_min, ), (double, v_max, ));
  DEF3(SetupAxisZoomConstraints, (ImAxis, axis, ), (double, z_min, ), (double, z_max, ));
  DEF4(SetupAxes, (ImString, x_label, ), (ImString, y_label, ), (ImPlotAxisFlags, x_flags, = 0), (ImPlotAxisFlags, y_flags, = 0));
  DEF5(SetupAxesLimits, (double, x_min, ), (double, x_max, ), (double, y_min, ), (double, y_max, ), (ImPlotCond, cond, = ImPlotCond_Once));
  DEF2(SetupLegend, (ImPlotLocation, location, ), (ImPlotLegendFlags, flags, = 0));
  DEF2(SetupMouseText, (ImPlotLocation, location, ), (ImPlotMouseTextFlags, flags, = 0));
  DEF0(SetupFinish);
  DEF4(SetNextAxisLimits, (ImAxis, axis, ), (double, v_min, ), (double, v_max, ), (ImPlotCond, cond, = ImPlotCond_Once));
  DEF3(SetNextAxisLinks, (ImAxis, axis, ), (double*, link_min, ), (double*, link_max, ));
  DEF1(SetNextAxisToFit, (ImAxis, axis, ));
  DEF5(SetNextAxesLimits, (double, x_min, ), (double, x_max, ), (double, y_min, ), (double, y_max, ), (ImPlotCond, cond, = ImPlotCond_Once));
  DEF0_F(SetNextAxesToFit, {
    return ImPlot::SetNextAxesToFit();
  });
  DEF6_F(PlotLine, (ImString, label_id, ), (std::vector<double>, xs, ), (std::vector<double>, ys, ), (ImPlotLineFlags, flags, = 0), (int, offset, = 0), (int, stride, = sizeof(double)), {
    ImPlotSpec spec;
    spec.Flags = flags;
    spec.Offset = offset;
    spec.Stride = stride;
    return ImPlot::PlotLine(label_id, xs.data(), ys.data(), xs.size(), spec);
  });
  DEF6_F(PlotScatter, (ImString, label_id, ), (std::vector<double>, xs, ), (std::vector<double>, ys, ), (ImPlotScatterFlags, flags, = 0), (int, offset, = 0), (int, stride, = sizeof(double)), {
    ImPlotSpec spec;
    spec.Flags = flags;
    spec.Offset = offset;
    spec.Stride = stride;
    return ImPlot::PlotScatter(label_id, xs.data(), ys.data(), xs.size(), spec);
  });
  DEF6_F(PlotStairs, (ImString, label_id, ), (std::vector<double>, xs, ), (std::vector<double>, ys, ), (ImPlotStairsFlags, flags, = 0), (int, offset, = 0), (int, stride, = sizeof(double)), {
    ImPlotSpec spec;
    spec.Flags = flags;
    spec.Offset = offset;
    spec.Stride = stride;
    return ImPlot::PlotStairs(label_id, xs.data(), ys.data(), xs.size(), spec);
  });
  DEF7_F(PlotShaded, (ImString, label_id, ), (std::vector<double>, xs, ), (std::vector<double>, ys, ), (double, yref, = 0), (ImPlotShadedFlags, flags, = 0), (int, offset, = 0), (int, stride, = sizeof(double)), {
    ImPlotSpec spec;
    spec.Flags = flags;
    spec.Offset = offset;
    spec.Stride = stride;
    return ImPlot::PlotShaded(label_id, xs.data(), ys.data(), xs.size(), yref, spec);
  });
  DEF7_F(PlotShaded, (ImString, label_id, ), (std::vector<double>, xs, ), (std::vector<double>, ys1, ), (std::vector<double>, ys2, ), (ImPlotShadedFlags, flags, = 0), (int, offset, = 0), (int, stride, = sizeof(double)), {
    ImPlotSpec spec;
    spec.Flags = flags;
    spec.Offset = offset;
    spec.Stride = stride;
    return ImPlot::PlotShaded(label_id, xs.data(), ys1.data(), ys2.data(), xs.size(), spec);
  });
  DEF7_F(PlotBars, (ImString, label_id, ), (std::vector<double>, xs, ), (std::vector<double>, ys, ), (double, bar_size, ), (ImPlotBarsFlags, flags, = 0), (int, offset, = 0), (int, stride, = sizeof(double)), {
    ImPlotSpec spec;
    spec.Flags = flags;
    spec.Offset = offset;
    spec.Stride = stride;
    return ImPlot::PlotBars(label_id, xs.data(), ys.data(), xs.size(), bar_size, spec);
  });
  DEF7_F(PlotErrorBars, (ImString, label_id, ), (std::vector<double>, xs, ), (std::vector<double>, ys, ), (std::vector<double>, err, ), (ImPlotErrorBarsFlags, flags, = 0), (int, offset, = 0), (int, stride, = sizeof(double)), {
    ImPlotSpec spec;
    spec.Flags = flags;
    spec.Offset = offset;
    spec.Stride = stride;
    return ImPlot::PlotErrorBars(label_id, xs.data(), ys.data(), err.data(), xs.size(), spec);
  });
  DEF8_F(PlotErrorBars, (ImString, label_id, ), (std::vector<double>, xs, ), (std::vector<double>, ys, ), (std::vector<double>, neg, ), (std::vector<double>, pos, ), (ImPlotErrorBarsFlags, flags, = 0), (int, offset, = 0), (int, stride, = sizeof(double)), {
    ImPlotSpec spec;
    spec.Flags = flags;
    spec.Offset = offset;
    spec.Stride = stride;
    return ImPlot::PlotErrorBars(label_id, xs.data(), ys.data(), neg.data(), pos.data(), xs.size(), spec);
  });
  DEF7_F(PlotStems, (ImString, label_id, ), (std::vector<double>, xs, ), (std::vector<double>, ys, ), (double, ref, = 0), (ImPlotStemsFlags, flags, = 0), (int, offset, = 0), (int, stride, = sizeof(double)), {
    ImPlotSpec spec;
    spec.Flags = flags;
    spec.Offset = offset;
    spec.Stride = stride;
    return ImPlot::PlotStems(label_id, xs.data(), ys.data(), xs.size(), ref, spec);
  });
  DEF6_F(PlotDigital, (ImString, label_id, ), (std::vector<double>, xs, ), (std::vector<double>, ys, ), (ImPlotDigitalFlags, flags, = 0), (int, offset, = 0), (int, stride, = sizeof(double)), {
    ImPlotSpec spec;
    spec.Flags = flags;
    spec.Offset = offset;
    spec.Stride = stride;
    return ImPlot::PlotDigital(label_id, xs.data(), ys.data(), xs.size(), spec);
  });
  DEF8_F(PlotImage, (ImString, label_id, ), (ImTextureRef, tex_ref, ), (const ImPlotPoint&, bounds_min, ), (const ImPlotPoint&, bounds_max, ), (const ImVec2&, uv0, = ImVec2_Zero), (const ImVec2&, uv1, = ImVec2_One), (const ImVec4&, tint_col, = ImVec4_One), (ImPlotImageFlags, flags, = 0), {
    ImPlotSpec spec;
    spec.Flags = flags;
    return ImPlot::PlotImage(label_id, tex_ref, bounds_min, bounds_max, uv0, uv1, tint_col, spec);
  });
  DEF5_F(PlotText, (ImString, text, ), (double, x, ), (double, y, ), (const ImVec2&, pix_offset, = ImVec2_Zero), (ImPlotTextFlags, flags, = 0), {
    ImPlotSpec spec;
    spec.Flags = flags;
    return ImPlot::PlotText(text, x, y, pix_offset, spec);
  });
  DEF2_F(PlotDummy, (ImString, label_id, ), (ImPlotDummyFlags, flags, = 0), {
    ImPlotSpec spec;
    spec.Flags = flags;
    return ImPlot::PlotDummy(label_id, spec);
  });
  DEF9(DragPoint, (int, id, ), (double*, x, ), (double*, y, ), (const ImVec4&, col, ), (float, size, = 4), (ImPlotDragToolFlags, flags, = 0), (bool*, out_clicked, = nullptr), (bool*, out_hovered, = nullptr), (bool*, out_held, = nullptr));
  DEF8(DragLineX, (int, id, ), (double*, x, ), (const ImVec4&, col, ), (float, thickness, = 1), (ImPlotDragToolFlags, flags, = 0), (bool*, out_clicked, = nullptr), (bool*, out_hovered, = nullptr), (bool*, out_held, = nullptr));
  DEF8(DragLineY, (int, id, ), (double*, y, ), (const ImVec4&, col, ), (float, thickness, = 1), (ImPlotDragToolFlags, flags, = 0), (bool*, out_clicked, = nullptr), (bool*, out_hovered, = nullptr), (bool*, out_held, = nullptr));
  DEF10(DragRect, (int, id, ), (double*, x1, ), (double*, y1, ), (double*, x2, ), (double*, y2, ), (const ImVec4&, col, ), (ImPlotDragToolFlags, flags, = 0), (bool*, out_clicked, = nullptr), (bool*, out_hovered, = nullptr), (bool*, out_held, = nullptr));
  DEF6(Annotation, (double, x, ), (double, y, ), (const ImVec4&, col, ), (const ImVec2&, pix_offset, ), (bool, clamp, ), (bool, round, = false));
  DEF6_F(Annotation, (double, x, ), (double, y, ), (const ImVec4&, col, ), (const ImVec2&, pix_offset, ), (bool, clamp, ), (ImString, txt, ), {
    return ImPlot::Annotation(x, y, col, pix_offset, clamp, "%s", txt);
  });
  DEF3(TagX, (double, x, ), (const ImVec4&, col, ), (bool, round, = false));
  DEF3_F(TagX, (double, x, ), (const ImVec4&, col, ), (ImString, txt, ), {
    return ImPlot::TagX(x, col, "%s", txt);
  });
  DEF3(TagY, (double, y, ), (const ImVec4&, col, ), (bool, round, = false));
  DEF3_F(TagY, (double, y, ), (const ImVec4&, col, ), (ImString, txt, ), {
    return ImPlot::TagY(y, col, "%s", txt);
  });
  DEF1(SetAxis, (ImAxis, axis, ));
  DEF2(SetAxes, (ImAxis, x_axis, ), (ImAxis, y_axis, ));
  DEF3(PixelsToPlot, (const ImVec2&, pix, ), (ImAxis, x_axis, = IMPLOT_AUTO), (ImAxis, y_axis, = IMPLOT_AUTO));
  DEF4(PixelsToPlot, (float, x, ), (float, y, ), (ImAxis, x_axis, = IMPLOT_AUTO), (ImAxis, y_axis, = IMPLOT_AUTO));
  DEF3(PlotToPixels, (const ImPlotPoint&, plt, ), (ImAxis, x_axis, = IMPLOT_AUTO), (ImAxis, y_axis, = IMPLOT_AUTO));
  DEF4(PlotToPixels, (double, x, ), (double, y, ), (ImAxis, x_axis, = IMPLOT_AUTO), (ImAxis, y_axis, = IMPLOT_AUTO));
  DEF0(GetPlotPos);
  DEF0(GetPlotSize);
  DEF2(GetPlotMousePos, (ImAxis, x_axis, = IMPLOT_AUTO), (ImAxis, y_axis, = IMPLOT_AUTO));
  DEF2(GetPlotLimits, (ImAxis, x_axis, = IMPLOT_AUTO), (ImAxis, y_axis, = IMPLOT_AUTO));
  DEF0(IsPlotHovered);
  DEF1(IsAxisHovered, (ImAxis, axis, ));
  DEF0(IsSubplotsHovered);
  DEF0(IsPlotSelected);
  DEF2(GetPlotSelection, (ImAxis, x_axis, = IMPLOT_AUTO), (ImAxis, y_axis, = IMPLOT_AUTO));
  DEF0(CancelPlotSelection);
  DEF2(HideNextItem, (bool, hidden, = true), (ImPlotCond, cond, = ImPlotCond_Once));
  DEF2(BeginAlignedPlots, (ImString, group_id, ), (bool, vertical, = true));
  DEF0(EndAlignedPlots);
  DEF2(BeginLegendPopup, (ImString, label_id, ), (ImGuiMouseButton, mouse_button, = 1));
  DEF0(EndLegendPopup);
  DEF1(IsLegendEntryHovered, (ImString, label_id, ));
  DEF0(BeginDragDropTargetPlot);
  DEF1(BeginDragDropTargetAxis, (ImAxis, axis, ));
  DEF0(BeginDragDropTargetLegend);
  DEF0(EndDragDropTarget);
  DEF1(BeginDragDropSourcePlot, (ImGuiDragDropFlags, flags, = 0));
  DEF2(BeginDragDropSourceAxis, (ImAxis, axis, ), (ImGuiDragDropFlags, flags, = 0));
  DEF2(BeginDragDropSourceItem, (ImString, label_id, ), (ImGuiDragDropFlags, flags, = 0));
  DEF0(EndDragDropSource);
  DEF2(PushStyleColor, (ImPlotCol, idx, ), (ImU32, col, ));
  DEF2(PushStyleColor, (ImPlotCol, idx, ), (const ImVec4&, col, ));
  DEF1(PopStyleColor, (int, count, = 1));
  DEF2(PushStyleVar, (ImPlotStyleVar, idx, ), (float, val, ));
  DEF2(PushStyleVar, (ImPlotStyleVar, idx, ), (int, val, ));
  DEF2(PushStyleVar, (ImPlotStyleVar, idx, ), (const ImVec2&, val, ));
  DEF1(PopStyleVar, (int, count, = 1));
  // four functions obsoleted in v1.0 of ImPlot:
  // SetNextLineStyle, SetNextFillStyle, SetNextMarkerStyle, SetNextErrorBarStyle
  DEF1(PushPlotClipRect, (float, expand, = 0));
  DEF0(PopPlotClipRect);
}

// NOLINTEND(whitespace/line_length)
