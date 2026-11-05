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

// Python bindings for MuJoCo platform UX components.

#include <array>
#include <string>
#include <tuple>
#include <vector>

#include <imgui.h>
#include <implot.h>
#include <mujoco/mujoco.h>
#include <mujoco/studio/sim/sim_history.h>
#include <mujoco/studio/sim/step_control.h>
#include <mujoco/studio/ux/console.h>
#include <mujoco/studio/ux/gui.h>
#include <mujoco/studio/ux/gui_helpers.h>
#include <mujoco/studio/ux/interaction.h>
#include <mujoco/studio/ux/picture_gui.h>
#include "specs_wrapper.h"
#include "structs.h"
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

namespace py = pybind11;

namespace {

struct UxState {
  // Read/edited by step_control_gui
  int speed_index = 0;

  // Read/edited by state_gui
  std::vector<mjtNum> state;
  int state_sig = 0;

  // Read/edited by watch_gui
  char watch_field_name[256] = {0};
  int watch_field_index = 0;

  // Read/edited by camera_selection_gui
  int camera_index = mujoco::studio::kTumbleCameraIdx;

  // Read/edited by view_gizmo_gui.
  mujoco::studio::ViewGizmoAnimation view_gizmo_animation;

  // Read/edited by simulation_gui.
  int key_idx = -1;
  int nthread = 0;
  bool update_threadpool = false;
  mujoco::studio::SimulationTimelineState timeline;
};

struct RenderFlags {
  std::array<uint8_t, mjNRNDFLAG> flags = {0};
};

}  // namespace

namespace mujoco::python::studio {

void InitUx(py::module_& m) {
  py::module_::import("mujoco._structs");
  py::class_<RenderFlags>(m, "RenderFlags")
      .def(py::init<>())
      .def_readwrite("flags", &RenderFlags::flags);

  m.doc() = "MuJoCo platform UX components.";

  py::enum_<mujoco::studio::GuiTheme>(m, "GuiTheme")
      .value("LIGHT", mujoco::studio::GuiTheme::kLight)
      .value("DARK", mujoco::studio::GuiTheme::kDark)
      .value("CLASSIC", mujoco::studio::GuiTheme::kClassic);

  py::class_<UxState>(m, "UxState")
      .def(py::init<>())
      .def_readwrite("speed_index", &UxState::speed_index)
      .def_readwrite("state", &UxState::state)
      .def_readwrite("state_sig", &UxState::state_sig)
      .def_readwrite("watch_field_index", &UxState::watch_field_index)
      .def_readwrite("camera_index", &UxState::camera_index)
      .def_readwrite("key_idx", &UxState::key_idx)
      .def_readwrite("nthread", &UxState::nthread)
      .def_readwrite("update_threadpool", &UxState::update_threadpool)
      .def_property(
          "sim_head_time",
          [](const UxState& self) { return self.timeline.sim_head_time; },
          [](UxState& self, double val) { self.timeline.sim_head_time = val; }
      )
      .def_property(
          "history_size",
          [](const UxState& self) { return self.timeline.history_size; },
          [](UxState& self, int val) { self.timeline.SetHistorySize(val); }
      )
      .def_property(
          "history_index",
          [](const UxState& self) { return self.timeline.history_index; },
          [](UxState& self, int val) { self.timeline.SetHistoryIndex(val); }
      )
      .def_property(
          "watch_field_name",
          [](const UxState& self) {
            return std::string(self.watch_field_name);
          },
          [](UxState& self, const std::string& val) {
            std::snprintf(self.watch_field_name, sizeof(self.watch_field_name),
                          "%s", val.c_str());
          });

  m.def(
      "setup_theme",
      [](mujoco::studio::GuiTheme theme) {
        py::gil_scoped_release no_gil;
        mujoco::studio::SetupTheme(theme);
      },
      py::arg("theme"), "Set up Dear ImGui visual theme.");

  m.def(
      "set_imgui_context",
      [](intptr_t ptr) {
        ImGui::SetCurrentContext(reinterpret_cast<ImGuiContext*>(ptr));
      },
      py::arg("ptr"), "Set ImGui context pointer.");

  // Each pybind extension module holds its own copy of the ImGui/ImPlot
  // globals (extension modules are loaded RTLD_LOCAL), so the viewer that
  // owns the contexts must share the pointers with this module. Without
  // this, the plotting GUIs would dereference a null ImPlot context.
  m.def(
      "set_implot_context",
      [](intptr_t ptr) {
        ImPlot::SetCurrentContext(reinterpret_cast<ImPlotContext*>(ptr));
      },
      py::arg("ptr"), "Set ImPlot context pointer.");

  m.def(
      "configure_docking_layout",
      [](bool show_toolbar, bool show_status_bar) {
        py::gil_scoped_release no_gil;
        ImVec4 r = mujoco::studio::ConfigureDockingLayout(show_toolbar,
                                                          show_status_bar);
        return std::make_tuple(r.x, r.y, r.z, r.w);
      },
      py::arg("show_toolbar") = true, py::arg("show_status_bar") = true,
      "Configure the docking layout with Options (left) and Inspector (right) "
      "panes. Returns (x, y, w, h) of the central workspace area.");

  m.def(
      "step_control_gui",
      [](mujoco::studio::StepControl* step_control, UxState& ux_state) {
        py::gil_scoped_release no_gil;
        mujoco::studio::StepControlGui(step_control, ux_state.speed_index);
      },
      py::arg("step_control"), py::arg("ux_state"),
      "Render the simulation stepping control GUI. Modifies "
      "ux_state.speed_index.");

  m.def(
      "reset_history",
      [](mujoco::studio::SimHistory* history, UxState& ux_state,
         py::object model_obj, py::object data_obj) {
        mjModel* model =
            py::cast<mujoco::python::MjModelWrapper&>(model_obj).get();
        mjData* data = py::cast<mujoco::python::MjDataWrapper&>(data_obj).get();
        py::gil_scoped_release no_gil;
        mujoco::studio::ResetHistory(*history, ux_state.timeline, model, data);
      },
      py::arg("history"), py::arg("ux_state"), py::arg("model"),
      py::arg("data"),
      "(Re)initialize the history buffer for the model, reset the timeline and "
      "record the current state as the first frame. Call on model load and "
      "after a reset.");

  m.def(
      "record_history",
      [](mujoco::studio::SimHistory* history, UxState& ux_state,
         py::object model_obj, py::object data_obj) {
        mjModel* model =
            py::cast<mujoco::python::MjModelWrapper&>(model_obj).get();
        mjData* data = py::cast<mujoco::python::MjDataWrapper&>(data_obj).get();
        py::gil_scoped_release no_gil;
        mujoco::studio::RecordHistoryFrame(*history, ux_state.timeline, model,
                                           data);
      },
      py::arg("history"), py::arg("ux_state"), py::arg("model"),
      py::arg("data"),
      "Append the current state to the history buffer and move the head of the "
      "timeline to it. Call whenever the simulation has advanced.");

  m.def(
      "load_history_frame",
      [](mujoco::studio::SimHistory* history, py::object model_obj,
         py::object data_obj, int index) {
        mjModel* model =
            py::cast<mujoco::python::MjModelWrapper&>(model_obj).get();
        mjData* data = py::cast<mujoco::python::MjDataWrapper&>(data_obj).get();
        py::gil_scoped_release no_gil;
        mujoco::studio::LoadHistoryFrame(*history, model, data, index);
      },
      py::arg("history"), py::arg("model"), py::arg("data"), py::arg("index"),
      "Load the history frame at `index` (0 is the most recent state) into "
      "`data`. A no-op if the frame is empty.");

  m.def(
      "simulation_gui",
      [](py::object model_obj, py::object data_obj,
         mujoco::studio::StepControl* step_control, UxState& ux_state,
         py::function reset, py::function reload, py::function align,
         py::function save_keyframe) {
        mjModel* model =
            py::cast<mujoco::python::MjModelWrapper&>(model_obj).get();
        mjData* data = py::cast<mujoco::python::MjDataWrapper&>(data_obj).get();
        // The GIL is held throughout: the callbacks call back into Python.
        mujoco::studio::SimulationGuiContext ctx;
        ctx.model = model;
        ctx.data = data;
        ctx.step_control = step_control;
        ctx.timeline = &ux_state.timeline;
        ctx.speed_index = &ux_state.speed_index;
        ctx.key_idx = &ux_state.key_idx;
        ctx.nthread = &ux_state.nthread;
        ctx.update_threadpool = &ux_state.update_threadpool;
        ctx.reset = [&reset]() { reset(); };
        ctx.reload = [&reload]() { reload(); };
        ctx.align = [&align]() { align(); };
        ctx.save_keyframe = [&save_keyframe]() { save_keyframe(); };
        mujoco::studio::SimulationGui(ctx);
      },
      py::arg("model"), py::arg("data"), py::arg("step_control"),
      py::arg("ux_state"), py::arg("reset"), py::arg("reload"),
      py::arg("align"), py::arg("save_keyframe"),
      "Render the full Simulation panel: reset/reload/align, run/pause, speed, "
      "the history scrubber, keyframes and thread count. The four callbacks "
      "are invoked for the corresponding buttons.");

  m.def(
      "timeline_scrubber_gui",
      [](py::object model_obj, py::object data_obj,
         mujoco::studio::StepControl* step_control, UxState& ux_state) {
        mjModel* model =
            py::cast<mujoco::python::MjModelWrapper&>(model_obj).get();
        mjData* data = py::cast<mujoco::python::MjDataWrapper&>(data_obj).get();
        py::gil_scoped_release no_gil;
        mujoco::studio::TimelineScrubberGui(model, data, *step_control,
                                            ux_state.timeline);
      },
      py::arg("model"), py::arg("data"), py::arg("step_control"),
      py::arg("ux_state"),
      "Render the timeline scrubber widget (spine + draggable knob).");

  m.def(
      "theme_select_gui",
      [](mujoco::studio::GuiTheme theme) {
        py::gil_scoped_release no_gil;
        bool changed = mujoco::studio::ThemeSelectGui(&theme);
        return std::make_tuple(changed, theme);
      },
      py::arg("theme"),
      "Render the GUI theme selector. Returns (changed, theme).");

  m.def(
      "theme_menu_gui",
      [](mujoco::studio::GuiTheme theme) {
        py::gil_scoped_release no_gil;
        bool changed = mujoco::studio::ThemeMenuGui(&theme);
        return std::make_tuple(changed, theme);
      },
      py::arg("theme"), "Render the Theme menu. Returns (changed, theme).");

  m.def("get_default_ini_path", &mujoco::studio::GetDefaultIniPath,
        "Returns the default INI configuration file path (~/.mujoco.ini).");

  m.def(
      "load_theme",
      [](const std::string& ini_path, mujoco::studio::GuiTheme def_theme) {
        py::gil_scoped_release no_gil;
        return mujoco::studio::LoadTheme(ini_path, def_theme);
      },
      py::arg("ini_path") = "",
      py::arg("def_theme") = mujoco::studio::GuiTheme::kLight,
      "Loads the GUI theme from an INI file.");

  m.def(
      "load_settings",
      [](const std::string& ini_path, mujoco::studio::GuiTheme def_theme) {
        py::gil_scoped_release no_gil;
        return mujoco::studio::LoadSettings(ini_path, def_theme);
      },
      py::arg("ini_path") = "",
      py::arg("def_theme") = mujoco::studio::GuiTheme::kLight,
      "Loads settings and theme from an INI file into ImGui.");

  m.def(
      "save_settings",
      [](mujoco::studio::GuiTheme theme, const std::string& ini_path) {
        py::gil_scoped_release no_gil;
        mujoco::studio::SaveSettings(theme, ini_path);
      },
      py::arg("theme"), py::arg("ini_path") = "",
      "Saves settings and theme to an INI file.");

  m.def(
      "reset_config",
      [](const std::string& ini_path) {
        py::gil_scoped_release no_gil;
        mujoco::studio::ResetConfig(ini_path);
      },
      py::arg("ini_path") = "", "Resets the configuration INI file.");

  m.def(
      "label_selection_gui",
      [](mujoco::python::MjvOptionWrapper& vis_options) {
        py::gil_scoped_release no_gil;
        return mujoco::studio::LabelSelectionGui(vis_options.get());
      },
      py::arg("vis_options"), "Render the visualization label selection GUI.");

  m.def(
      "frame_selection_gui",
      [](mujoco::python::MjvOptionWrapper& vis_options) {
        py::gil_scoped_release no_gil;
        return mujoco::studio::FrameSelectionGui(vis_options.get());
      },
      py::arg("vis_options"), "Render the visualization frame selection GUI.");

  m.def(
      "camera_selection_gui",
      [](const mujoco::python::MjModelWrapper& model,
         mujoco::python::MjDataWrapper& data,
         mujoco::python::MjvCameraWrapper& camera, UxState& ux_state) {
        py::gil_scoped_release no_gil;
        bool changed = mujoco::studio::CameraSelectionGui(
            model.get(), data.get(), *camera.get(), ux_state.camera_index);
        return changed;
      },
      py::arg("model"), py::arg("data"), py::arg("camera"), py::arg("ux_state"),
      "Render the camera selection GUI. Modifies ux_state.camera_index. "
      "Returns true if camera changed.");

  m.def(
      "view_gizmo_gui",
      [](const mujoco::python::MjModelWrapper& model,
         mujoco::python::MjDataWrapper& data,
         mujoco::python::MjvCameraWrapper& camera, UxState& ux_state,
         const std::array<float, 4>& workspace_rect) {
        py::gil_scoped_release no_gil;
        return mujoco::studio::ViewGizmoGui(
            model.get(), data.get(), *camera.get(), ux_state.camera_index,
            ux_state.view_gizmo_animation,
            ImVec4(workspace_rect[0], workspace_rect[1], workspace_rect[2],
                   workspace_rect[3]));
      },
      py::arg("model"), py::arg("data"), py::arg("camera"), py::arg("ux_state"),
      py::arg("workspace_rect"),
      "Renders the XYZ view gizmo in workspace_rect, the (x, y, w, h) from "
      "configure_docking_layout. May modify ux_state.camera_index. Returns "
      "true if the camera changed.");

  m.def(
      "set_camera",
      [](const mujoco::python::MjModelWrapper& model,
         mujoco::python::MjvCameraWrapper& camera, int request_idx) {
        py::gil_scoped_release no_gil;
        return mujoco::studio::SetCamera(model.get(), camera.get(),
                                         request_idx);
      },
      py::arg("model"), py::arg("camera"), py::arg("request_idx"),
      "Set the camera index and update the camera object.");

  m.def(
      "set_speed_index",
      [](mujoco::studio::StepControl* step_control, int speed_index,
         int request_idx) {
        {
          py::gil_scoped_release no_gil;
          mujoco::studio::SetSpeedIndex(step_control, speed_index, request_idx);
        }
        return speed_index;
      },
      py::arg("step_control"), py::arg("speed_index"), py::arg("request_idx"),
      "Set the simulation speed index. Returns new_speed_index.");

  m.def(
      "physics_gui",
      [](mujoco::python::MjModelWrapper& model, mujoco::python::MjSpec* spec,
         float min_width) {
        py::gil_scoped_release no_gil;
        mujoco::studio::PhysicsGui(model.get(), spec ? spec->ptr : nullptr,
                                   min_width);
      },
      py::arg("model"), py::arg("spec") = nullptr,
      py::arg("min_width") = 150.0f, "Render the physics settings UI.");

  m.def(
      "rendering_gui",
      [](const mujoco::python::MjModelWrapper& model,
         mujoco::python::MjvOptionWrapper& vis_options,
         RenderFlags& render_flags) {
        py::gil_scoped_release no_gil;
        mjtByte flags[mjNRNDFLAG] = {0};
        for (int i = 0; i < mjNRNDFLAG; ++i) {
          flags[i] = render_flags.flags[i];
        }
        mujoco::studio::RenderingGui(model.get(), vis_options.get(), flags,
                                     150.0f);
        for (int i = 0; i < mjNRNDFLAG; ++i) {
          render_flags.flags[i] = flags[i];
        }
      },
      py::arg("model"), py::arg("vis_options"), py::arg("render_flags"),
      "Render the rendering settings UI. Modifies render_flags in place.");

  m.def(
      "groups_gui",
      [](const mujoco::python::MjModelWrapper& model,
         mujoco::python::MjvOptionWrapper& vis_options, float min_width) {
        py::gil_scoped_release no_gil;
        mujoco::studio::GroupsGui(model.get(), vis_options.get(), min_width);
      },
      py::arg("model"), py::arg("vis_options"), py::arg("min_width") = 150.0f,
      "Render the visibility groups UI.");

  m.def(
      "visualization_gui",
      [](mujoco::python::MjModelWrapper& model,
         mujoco::python::MjvOptionWrapper& vis_options,
         mujoco::python::MjvCameraWrapper& camera, float min_width) {
        py::gil_scoped_release no_gil;
        mujoco::studio::VisualizationGui(model.get(), vis_options.get(),
                                         camera.get(), min_width);
      },
      py::arg("model"), py::arg("vis_options"), py::arg("camera"),
      py::arg("min_width") = 150.0f, "Render the visualization settings UI.");

  m.def(
      "controls_gui",
      [](const mujoco::python::MjModelWrapper& model,
         mujoco::python::MjDataWrapper& data,
         mujoco::python::MjvOptionWrapper& vis_options) {
        py::gil_scoped_release no_gil;
        mujoco::studio::ControlsGui(model.get(), data.get(), vis_options.get());
      },
      py::arg("model"), py::arg("data"), py::arg("vis_options"),
      "Render the actuator controls UI.");

  m.def(
      "joints_gui",
      [](const mujoco::python::MjModelWrapper& model,
         mujoco::python::MjDataWrapper& data,
         mujoco::python::MjvOptionWrapper& vis_options) {
        py::gil_scoped_release no_gil;
        mujoco::studio::JointsGui(model.get(), data.get(), vis_options.get());
      },
      py::arg("model"), py::arg("data"), py::arg("vis_options"),
      "Render the joints UI.");

  m.def(
      "sensor_gui",
      [](const mujoco::python::MjModelWrapper& model,
         mujoco::python::MjDataWrapper& data) {
        py::gil_scoped_release no_gil;
        mujoco::studio::SensorGui(model.get(), data.get());
      },
      py::arg("model"), py::arg("data"), "Render the sensor data plot.");

  m.def(
      "state_gui",
      [](const mujoco::python::MjModelWrapper& model,
         mujoco::python::MjDataWrapper& data, UxState& ux_state,
         float min_width) {
        py::gil_scoped_release no_gil;
        mujoco::studio::StateGui(model.get(), data.get(), ux_state.state,
                                 ux_state.state_sig, min_width);
      },
      py::arg("model"), py::arg("data"), py::arg("ux_state"),
      py::arg("min_width") = 150.0f,
      "Render the state UI. Modifies ux_state.state and ux_state.state_sig.");

  m.def(
      "watch_gui",
      [](const mujoco::python::MjModelWrapper& model,
         mujoco::python::MjDataWrapper& data, UxState& ux_state) {
        py::gil_scoped_release no_gil;
        mujoco::studio::WatchGui(
            model.get(), data.get(), ux_state.watch_field_name,
            sizeof(ux_state.watch_field_name), ux_state.watch_field_index);
      },
      py::arg("model"), py::arg("data"), py::arg("ux_state"),
      "Render the watch UI. Modifies ux_state.watch_field_name and "
      "ux_state.watch_field_index.");

  m.def(
      "noise_gui",
      [](mujoco::studio::StepControl* step_control) {
        py::gil_scoped_release no_gil;
        mujoco::studio::NoiseGui(step_control);
      },
      py::arg("step_control"), "Render the noise UI.");

  m.def(
      "convergence_gui",
      [](const mujoco::python::MjModelWrapper& model,
         mujoco::python::MjDataWrapper& data) {
        py::gil_scoped_release no_gil;
        mujoco::studio::ConvergenceGui(model.get(), data.get());
      },
      py::arg("model"), py::arg("data"),
      "Render the solver convergence chart.");

  m.def(
      "counts_gui",
      [](const mujoco::python::MjModelWrapper& model,
         mujoco::python::MjDataWrapper& data) {
        py::gil_scoped_release no_gil;
        mujoco::studio::CountsGui(model.get(), data.get());
      },
      py::arg("model"), py::arg("data"), "Render the solver counts chart.");

  m.def(
      "info_gui",
      [](const mujoco::python::MjModelWrapper& model,
         mujoco::python::MjDataWrapper& data, bool paused, float fps,
         float gpu_ms) {
        py::gil_scoped_release no_gil;
        mujoco::studio::InfoGui(model.get(), data.get(), paused, fps, gpu_ms);
      },
      py::arg("model"), py::arg("data"), py::arg("paused"), py::arg("fps"),
      py::arg("gpu_ms") = 0.0f, "Render the simulation info UI.");

  m.def(
      "init_console",
      []() {
        py::gil_scoped_release no_gil;
        mujoco::studio::Console::Init();
      },
      "Installs a MuJoCo log handler that adds messages to the Studio "
      "Console and passes them on to the previous handler. Only the first "
      "call installs it.");

  m.def(
      "clear_console",
      []() {
        py::gil_scoped_release no_gil;
        mujoco::studio::Console::Get().Clear();
      },
      "Clears all messages and resets counts in the Studio Console.");

  m.def(
      "add_console_message",
      [](int level, const std::string& message) {
        py::gil_scoped_release no_gil;
        mujoco::studio::Console::Get().Add(static_cast<mjtLogLevel>(level),
                                           message);
      },
      py::arg("level"), py::arg("message"),
      "Adds a message at the given mjtLogLevel to the Studio Console (not "
      "printed).");

  py::class_<mujoco::studio::ConsoleCounts>(m, "ConsoleCounts")
      .def_readonly("messages", &mujoco::studio::ConsoleCounts::messages)
      .def_readonly("warnings", &mujoco::studio::ConsoleCounts::warnings)
      .def_readonly("errors", &mujoco::studio::ConsoleCounts::errors);

  m.def(
      "get_console_counts",
      []() {
        py::gil_scoped_release no_gil;
        return mujoco::studio::Console::Get().GetCounts();
      },
      "Returns ConsoleCounts (messages, warnings, errors) for the Studio "
      "Console.");

  m.def(
      "get_console_lines",
      []() {
        py::gil_scoped_release no_gil;
        return mujoco::studio::Console::Get().GetLines();
      },
      "Returns a list of log lines currently in the Studio Console. It "
      "copies every line, so call it on demand, e.g. in tests and scripts, "
      "rather than every frame.");

  m.def(
      "console_gui",
      []() {
        py::gil_scoped_release no_gil;
        mujoco::studio::Console::Get().WindowGui();
      },
      "Renders the Studio Console table.");

  m.def(
      "console_status_gui",
      [](bool show_console) {
        py::gil_scoped_release no_gil;
        mujoco::studio::Console::Get().StatusGui(&show_console);
        return show_console;
      },
      py::arg("show_console"),
      "Renders the Studio Console status bar summary. Returns updated "
      "show_console state.");

  m.attr("FREE_CAMERA_IDX") = mujoco::studio::kFreeCameraIdx;
  m.attr("TUMBLE_CAMERA_IDX") = mujoco::studio::kTumbleCameraIdx;
  m.attr("TRACKING_CAMERA_IDX") = mujoco::studio::kTrackingCameraIdx;

  py::enum_<mujoco::studio::CameraMotion>(m, "CameraMotion")
      .value("ZOOM", mujoco::studio::CameraMotion::ZOOM)
      .value("ORBIT", mujoco::studio::CameraMotion::ORBIT)
      .value("TRUCK_PEDESTAL", mujoco::studio::CameraMotion::TRUCK_PEDESTAL)
      .value("TRUCK_DOLLY", mujoco::studio::CameraMotion::TRUCK_DOLLY)
      .value("PAN_TILT", mujoco::studio::CameraMotion::PAN_TILT)
      .value("PLANAR_MOVE_H", mujoco::studio::CameraMotion::PLANAR_MOVE_H)
      .value("PLANAR_MOVE_V", mujoco::studio::CameraMotion::PLANAR_MOVE_V)
      .export_values();

  m.def(
      "MoveCamera",
      [](const mujoco::python::MjModelWrapper& model,
         const mujoco::python::MjDataWrapper& data,
         mujoco::python::MjvCameraWrapper& cam,
         mujoco::studio::CameraMotion motion, mjtNum dx, mjtNum dy) {
        py::gil_scoped_release no_gil;
        mujoco::studio::MoveCamera(model.get(), data.get(), cam.get(), motion,
                                   dx, dy);
      },
      py::arg("model"), py::arg("data"), py::arg("cam"), py::arg("motion"),
      py::arg("dx"), py::arg("dy"), "Moves the given camera.");

  m.def(
      "InitPerturb",
      [](const mujoco::python::MjModelWrapper& model,
         const mujoco::python::MjDataWrapper& data,
         const mujoco::python::MjvCameraWrapper& cam,
         mujoco::python::MjvPerturbWrapper& pert, int active) {
        py::gil_scoped_release no_gil;
        mujoco::studio::InitPerturb(model.get(), data.get(), cam.get(),
                                    pert.get(),
                                    static_cast<mjtPertBit>(active));
      },
      py::arg("model"), py::arg("data"), py::arg("cam"), py::arg("pert"),
      py::arg("active"), "Initializes mouse perturbation.");

  m.def(
      "MovePerturb",
      [](const mujoco::python::MjModelWrapper& model,
         const mujoco::python::MjDataWrapper& data,
         const mujoco::python::MjvCameraWrapper& cam,
         mujoco::python::MjvPerturbWrapper& pert, int action, mjtNum reldx,
         mjtNum reldy) {
        py::gil_scoped_release no_gil;
        mujoco::studio::MovePerturb(model.get(), data.get(), cam.get(),
                                    pert.get(), static_cast<mjtMouse>(action),
                                    reldx, reldy);
      },
      py::arg("model"), py::arg("data"), py::arg("cam"), py::arg("pert"),
      py::arg("action"), py::arg("reldx"), py::arg("reldy"),
      "Moves mouse perturbation.");

  py::class_<mujoco::studio::PickResult>(m, "PickResult")
      .def_readwrite("dist", &mujoco::studio::PickResult::dist)
      .def_readwrite("body", &mujoco::studio::PickResult::body)
      .def_readwrite("geom", &mujoco::studio::PickResult::geom)
      .def_readwrite("flex", &mujoco::studio::PickResult::flex)
      .def_readwrite("skin", &mujoco::studio::PickResult::skin)
      .def_property(
          "point",
          [](const mujoco::studio::PickResult& res) {
            return py::make_tuple(res.point[0], res.point[1], res.point[2]);
          },
          [](mujoco::studio::PickResult& res, const py::tuple& t) {
            res.point[0] = t[0].cast<mjtNum>();
            res.point[1] = t[1].cast<mjtNum>();
            res.point[2] = t[2].cast<mjtNum>();
          });

  m.def(
      "Pick",
      [](const mujoco::python::MjModelWrapper& model,
         const mujoco::python::MjDataWrapper& data,
         const mujoco::python::MjvCameraWrapper& cam, float x, float y,
         float aspect_ratio, const mujoco::python::MjvOptionWrapper& opt) {
        py::gil_scoped_release no_gil;
        return mujoco::studio::Pick(model.get(), data.get(), cam.get(), x, y,
                                    aspect_ratio, opt.get());
      },
      py::arg("model"), py::arg("data"), py::arg("cam"), py::arg("x"),
      py::arg("y"), py::arg("aspect_ratio"), py::arg("opt"),
      "Picks object under cursor.");

  m.def(
      "camera_to_string",
      [](const mujoco::python::MjDataWrapper& data,
         const mujoco::python::MjvCameraWrapper& camera) {
        py::gil_scoped_release no_gil;
        return mujoco::studio::CameraToString(data.get(), camera.get());
      },
      py::arg("data"), py::arg("camera"),
      "Returns an XML string representation of the camera.");

  // Picture-in-picture state and GUI binding for Python viewer plugins.
  py::class_<mujoco::studio::PipSource>(m, "PipSource")
      .def(py::init<std::string, ImTextureID, float>(), py::arg("name") = "",
           py::arg("texture") = ImTextureID_Invalid,
           py::arg("aspect_ratio") = 1.0f)
      .def_readwrite("name", &mujoco::studio::PipSource::name)
      .def_readwrite("texture", &mujoco::studio::PipSource::texture)
      .def_readwrite("aspect_ratio", &mujoco::studio::PipSource::aspect_ratio);

  py::class_<mujoco::studio::PipState>(m, "PipState")
      .def(py::init<>())
      .def_readwrite("camera", &mujoco::studio::PipState::camera)
      .def_readwrite("texture", &mujoco::studio::PipState::texture);

  m.def(
      "pip_gui",
      [](const std::vector<mujoco::studio::PipSource>& sources,
         std::vector<mujoco::studio::PipState> pips) {
        py::gil_scoped_release no_gil;
        mujoco::studio::PipGui(sources, &pips);
        return pips;
      },
      py::arg("sources"), py::arg("pips"));
}

}  // namespace mujoco::python::studio

#ifndef MUJOCO_STUDIO_COMBINED_MODULE
PYBIND11_MODULE(ux, m, pybind11::mod_gil_not_used()) {
  mujoco::python::studio::InitUx(m);
}
#endif
