# Copyright 2026 DeepMind Technologies Limited
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Python implementation of the default Studio Viewer application UI.

The class can be used as a plugin to run the full Studio Viewer application in
Python.

Architecture:
  App provides the full default UI/UX as a viewer-side plugin. This
  class is viewer-agnostic and as such does not own camera, vis_options, or
  perturb objects (these are provided by the viewer).

  Viewer classes (NativeViewer and WebViewer) own the renderer, camera,
  vis_options, and local deep-copied model/data used for rendering. The sim side
  owns the physical simulation and pushes state snapshots to the viewer via
  ViewerHandle.sync().
"""

from collections.abc import Iterator
import contextlib
import copy
import dataclasses
import logging
import re
import sys
import threading
import traceback
from typing import Any
import warnings

import mujoco
from mujoco.studio import messages
from mujoco.studio import plugin
from mujoco.studio import sim
from mujoco.studio import sim_history
from mujoco.studio import step_control
from mujoco.studio import ux
from mujoco.studio import viewer_utils
from mujoco.studio.app import inputs

from mujoco.studio.dear_imgui import dear_imgui as imgui


class _StudioLogHandler(logging.Handler):
  """Adds Python log records from any thread to the Studio Console.

  The records are only added to the console. Python's other logging handlers
  still print them, or logging.lastResort if there are none.
  """

  def emit(self, record: logging.LogRecord) -> None:
    try:
      msg = self.format(record)
      if record.levelno >= logging.ERROR:
        level = int(mujoco.mjtLogLevel.mjLOG_ERROR)
      elif record.levelno >= logging.WARNING:
        level = int(mujoco.mjtLogLevel.mjLOG_WARNING)
      elif record.levelno >= logging.INFO:
        level = int(mujoco.mjtLogLevel.mjLOG_INFO)
      else:
        level = int(mujoco.mjtLogLevel.mjLOG_DEBUG)
      ux.add_console_message(level, msg)
      # Python prints records that reach no handler with logging.lastResort.
      # This handler counts as one, so print records that reach no other
      # handler here, whatever their level, as they are in the console.
      last_resort = logging.lastResort
      if last_resort is not None and not self._reaches_other_handler(record):
        last_resort.handle(record)
    except Exception:  # pylint: disable=broad-exception-caught
      self.handleError(record)

  def _reaches_other_handler(self, record: logging.LogRecord) -> bool:
    """Returns whether the record also reaches a handler other than this one."""
    logger: logging.Logger | None = logging.getLogger(record.name)
    while logger is not None:
      if any(handler is not self for handler in logger.handlers):
        return True
      logger = logger.parent if logger.propagate else None
    return False


class _ConsoleHooks:
  """Adds Python warnings and uncaught exceptions to the Studio Console.

  Each hook adds the warning or exception to the console and then calls the
  hook it replaced, which prints it as before.
  """

  def __init__(self) -> None:
    self.orig_showwarning = warnings.showwarning
    self.orig_threading_excepthook = threading.excepthook
    self.orig_sys_excepthook = sys.excepthook

  def showwarning(
      self,
      message: Warning | str,
      category: type[Warning],
      filename: str,
      lineno: int,
      file: Any = None,
      line: str | None = None,
  ) -> None:
    """Forwards warnings to the Studio Console and original hook."""
    if file is None or file is sys.stderr:
      ux.add_console_message(
          int(mujoco.mjtLogLevel.mjLOG_WARNING),
          f'{filename}:{lineno}: {category.__name__}: {message}'.rstrip('\n'),
      )
    self.orig_showwarning(message, category, filename, lineno, file, line)

  def threading_excepthook(self, args: threading.ExceptHookArgs) -> None:
    """Forwards thread exceptions to the Studio Console and original hook."""
    if args.exc_type is not SystemExit:
      thread_name = args.thread.name if args.thread else 'unknown'
      tb = ''.join(
          traceback.format_exception(
              args.exc_type, args.exc_value, args.exc_traceback
          )
      ).rstrip('\n')
      ux.add_console_message(
          int(mujoco.mjtLogLevel.mjLOG_ERROR),
          f'Unhandled exception in thread {thread_name!r}:\n{tb}',
      )
    self.orig_threading_excepthook(args)

  def sys_excepthook(
      self,
      exc_type: type[BaseException],
      exc_value: BaseException,
      exc_traceback: Any,
  ) -> None:
    if not issubclass(exc_type, KeyboardInterrupt):
      tb = ''.join(
          traceback.format_exception(exc_type, exc_value, exc_traceback)
      ).rstrip('\n')
      ux.add_console_message(int(mujoco.mjtLogLevel.mjLOG_ERROR), tb)
    self.orig_sys_excepthook(exc_type, exc_value, exc_traceback)


_STUDIO_LOG_HANDLER = _StudioLogHandler()
_HOOKS_LOCK = threading.Lock()
_hooks: _ConsoleHooks | None = None


def install_console_hooks() -> None:
  """Adds logs, warnings, and uncaught exceptions to the Studio Console.

  Installs the MuJoCo log handler (see ux.init_console), adds a handler for
  Python log records, and hooks warnings.showwarning, threading.excepthook, and
  sys.excepthook. Each hook adds to the console and then calls what it replaced
  (falling back to logging.lastResort for log records that reach no other
  handler), leaving terminal and file output unchanged. Only the first call
  installs these; later calls only check again how logging is configured, e.g.
  whether web_server has configured its logger since.
  """
  global _hooks
  with _HOOKS_LOCK:
    if _hooks is None:
      ux.init_console()
      logging.getLogger().addHandler(_STUDIO_LOG_HANDLER)
      # MjSpec.compile() reports compiler warnings with warnings.warn(), which
      # attributes them to the parser. Show them on every load, not just the
      # first time, so that reloading a model shows its warnings again.
      warnings.filterwarnings(
          'always',
          category=UserWarning,
          module=re.escape(messages.__name__.rpartition('.')[0] + '.parser'),
      )
      _hooks = _ConsoleHooks()
      warnings.showwarning = _hooks.showwarning
      threading.excepthook = _hooks.threading_excepthook
      sys.excepthook = _hooks.sys_excepthook

    # When web_server configures its logger, it stops it propagating to the
    # root logger, so that logger needs the handler too.
    web_logger = logging.getLogger('WebServer')
    if (
        not web_logger.propagate
        and _STUDIO_LOG_HANDLER not in web_logger.handlers
    ):
      web_logger.addHandler(_STUDIO_LOG_HANDLER)


@dataclasses.dataclass(frozen=True)
class InitEvent(messages.Event):
  """Lifecycle event dispatched once when the studio_app App is initialised.

  Plugins that need access to the studio_app App should handle this event
  and cache the reference.
  """

  app: 'App'


@dataclasses.dataclass
class Config:
  """Configures the studio_app App.

  Pass an instance to App to choose what is visible on startup. The App keeps
  the instance as its live UI state, so the View menu and the keyboard
  shortcuts toggle these same fields.
  """

  show_info: bool = False
  show_profiler: bool = False
  show_console: bool = False
  show_options: bool = True
  show_toolbar: bool = False
  show_view_gizmo: bool = True
  show_status_bar: bool = True
  show_inspector: bool = True


# Sim-side plugins that back App's step control and history scrubbing UI.
# Unpack alongside App() in plugins=[...], or pass individual plugins when
# replacing one (e.g. a custom StepEvent plugin).
SIM_PLUGINS: tuple[plugin.SimPlugin, ...] = (
    step_control.StepControl(),
    sim_history.SimHistory(),
)


class App(plugin.ViewerPlugin):
  """Wraps a Viewer and adds Studio UI/UX.

  Pair with ``*SIM_PLUGINS`` in ``plugins=[...]`` for the default sim-side
  real-time CPU stepping and history scrubbing.
  """

  @property
  def model(self) -> mujoco.MjModel:
    return self.viewer.model

  @model.setter
  def model(self, value: mujoco.MjModel) -> None:
    self.viewer.model = value

  @property
  def model_path(self) -> str:
    return self.viewer.model_path

  @model_path.setter
  def model_path(self, value: str) -> None:
    self.viewer.model_path = value

  @property
  def data(self) -> mujoco.MjData:
    return self.viewer.data

  @data.setter
  def data(self, value: mujoco.MjData) -> None:
    self.viewer.data = value

  def __init__(self, config: Config | None = None) -> None:
    install_console_hooks()
    self._model_path: str | None = None
    self.theme = ux.load_theme()
    self.config = config or Config()
    self.status = 'Ready'
    self._reset_app_state()

  @messages.handler(priority=messages.Priority.CRITICAL)
  def _on_viewer_init(self, _: plugin.ViewerInitEvent) -> None:
    # Creating the web viewer configures the web server's logger.
    install_console_hooks()
    self.theme = ux.load_settings(def_theme=self.theme)
    ux.setup_theme(self.theme)
    self.align_camera()
    self.viewer.dispatch(InitEvent(app=self))

  def _reset_app_state(self) -> None:
    """Resets studio_app state (step control, ux)."""
    self.step_control_state = sim.StepControl()
    self.ux_state = ux.UxState()
    if self._viewer is not None:
      self.align_camera()

  def _reload_app_state(self, model: mujoco.MjModel) -> None:
    """Reloads studio_app state, keeping step control, speed, and camera."""
    old = self.ux_state
    self.ux_state = ux.UxState()
    self.ux_state.speed_index = old.speed_index
    camera = self.viewer.camera
    tracks_missing_body = (
        camera.type == int(mujoco.mjtCamera.mjCAMERA_TRACKING)
        and camera.trackbodyid >= model.nbody
    )
    if old.camera_index < model.ncam and not tracks_missing_body:
      self.ux_state.camera_index = old.camera_index
    else:
      self.align_camera()

  @contextlib.contextmanager
  def _send_if_changes(self) -> Iterator[None]:
    """Sends snapshots/events for any app or model config changed in the block."""
    with self.viewer.send_if_changes(
        lambda: messages.StepControlSnapshot.from_step_control(
            self.step_control_state
        ),
        lambda: sim_history.SimHistorySnapshot.from_ux_state(self.ux_state),
        lambda: messages.MjOptionSnapshot(opt=copy.deepcopy(self.model.opt)),
        lambda: messages.ThreadPoolEvent(nthread=self.ux_state.nthread),
    ):
      yield

  def align_camera(self) -> None:
    """Recenter the camera on the model's home camera, else the free camera."""
    cam_id = self.model.vis.global_.cameraid
    if 0 <= cam_id < self.model.ncam:
      self.ux_state.camera_index = ux.set_camera(
          self.model, self.viewer.camera, cam_id
      )
    else:
      mujoco.mjv_defaultFreeCamera(self.model, self.viewer.camera)
      if self.ux_state.camera_index not in (
          ux.TUMBLE_CAMERA_IDX,
          ux.FREE_CAMERA_IDX,
      ):
        self.ux_state.camera_index = ux.TUMBLE_CAMERA_IDX

  def reload_model(self) -> None:
    """Asks the sim to reload the current model."""
    self.viewer.send_to_sim(
        messages.LoadModelEvent(path=self.viewer.model_path)
    )

  def close(self) -> None:
    self.viewer.dispatch(messages.ExitEvent())

  def _forward_single_step(self) -> None:
    """Forwards a pending single-step request to the sim as an event."""
    if self.step_control_state.consume_single_step_request():
      self.viewer.send_to_sim(messages.SingleStepEvent())

  def handle_keyboard_events(self) -> None:
    """Handles keyboard events."""

    is_freecam_wasd = self.ux_state.camera_index == ux.FREE_CAMERA_IDX
    if inputs.handle_step_control_keyboard_events(
        self.step_control_state, self.ux_state
    ):
      return

    if inputs.handle_sim_history_keyboard_events(
        self.step_control_state, self.ux_state
    ):
      return

    if inputs.handle_reset_keyboard_events(self.model, self.data):
      self.reset_physics()
      return

    if inputs.handle_camera_select_keyboard_events(
        self.model, self.viewer.camera, self.ux_state
    ):
      return

    if inputs.handle_vis_options_keyboard_events(
        self.viewer.vis_options, is_freecam_wasd
    ):
      return

    if inputs.handle_miscellaneous_keyboard_events(self):
      return

    if is_freecam_wasd and self.viewer.client_camera is None:
      handled, self.viewer.cam_speed = (
          inputs.handle_freecam_wasd_keyboard_events(
              self.model, self.data, self.viewer.camera, self.viewer.cam_speed
          )
      )
      if handled:
        return

  def handle_camera_tracking_mouse_events(self) -> None:
    """Handles mouse events for camera tracking."""
    return inputs.handle_camera_tracking_mouse_events(
        self.model,
        self.data,
        self.viewer.camera,
        self.viewer.vis_options,
        self.ux_state,
    )

  def handle_mouse_events(
      self,
  ) -> None:
    """Handles mouse events."""
    return inputs.handle_mouse_events(
        self.model,
        self.data,
        self.viewer.camera,
        self.viewer.vis_options,
        self.viewer.perturb,
        self.ux_state,
        move_camera=self.viewer.client_camera is None,
    )

  def reset_physics(self) -> None:
    """Reset the physics."""
    if self.model is not None and self.data is not None:
      key_idx = self.ux_state.key_idx
      if key_idx >= self.model.nkey or key_idx < -1:
        key_idx = -1
        self.ux_state.key_idx = -1

      if key_idx >= 0:
        mujoco.mj_resetDataKeyframe(self.model, self.data, key_idx)
      else:
        mujoco.mj_resetData(self.model, self.data)
      mujoco.mj_forward(self.model, self.data)
      self.ux_state.sim_head_time = self.data.time
      # The viewer's mirror of the sim history is stale until the sim reports
      # its post-reset history. Setting the size to 0 also clamps the index.
      self.ux_state.history_size = 0
      self.viewer.send_to_sim(messages.ResetEvent(key=key_idx))
      # Discard any pre-reset snapshots so we don't overwrite the reset state.
      self.viewer.get_sim_snapshots()

  def save_keyframe(self) -> None:
    """Save the active state to the selected keyframe and notify the sim."""
    if self.model is not None and self.data is not None:
      key_idx = self.ux_state.key_idx
      if 0 <= key_idx < self.model.nkey:
        mujoco.mj_setKeyframe(self.model, self.data, key_idx)
        event = messages.KeyframeEvent.from_model(self.model, key_idx)
        if event is not None:
          self.viewer.send_to_sim(event)

  def apply_perturb(self) -> None:
    """Apply perturbation the model."""
    is_paused = (
        self.step_control_state.get_pause_state()
        == sim.PauseState.NORMAL_PAUSED
    )
    viewer_utils.apply_perturb(self.viewer, self.model, self.data, is_paused)

  def reset_physics_gui(self) -> None:
    """GUI to Reset the physics i.e., the reset button."""
    button_size = imgui.GetFrameHeight()
    square_size = imgui.Vec2(button_size, button_size)
    icon_reset_model = '\uf0e2'  # FontAwesome "undo" icon.
    if imgui.Button(icon_reset_model, square_size):
      self.reset_physics()
    imgui.SetItemTooltip('Reset')

  def is_running(self) -> bool:
    """Returns True if the application should continue running."""
    return self.viewer.is_running()

  def update(self) -> None:
    """Update the simulation and handle user input.

    Handles mouse input to compute perturbations or camera motion.
    Handles keyboard input e.g., for keybindings or camera motion.
    Applies the perturbations and advances the physics.
    """
    # A dropped file is loaded on the sim thread by the ViewerHandle, which
    # publishes the new model back to the viewer as a ModelEvent.
    drop_file = self.viewer.get_drop_file()
    if drop_file:
      self.viewer.send_to_sim(messages.LoadModelEvent(path=drop_file))

    with self._send_if_changes():
      # Handle user input.
      self.handle_mouse_events()
      self.handle_keyboard_events()

      # Apply perturbation forces from the viewer.
      self.apply_perturb()

      # Forward single-step requests as events (reliable, never dropped).
      self._forward_single_step()

  def build_gui(self) -> None:
    """Emit full Studio UI."""
    # TODO(matijak): Consider adding a PreBuildGuiEvent and moving the theme and
    # dockspace setup below into a handler for it. That would guarantee the
    # dockspace exists before any plugin's BuildGuiEvent handler submits
    # windows, instead of relying on App's Priority.CRITICAL.
    ux.setup_theme(self.theme)
    workspace_rect = ux.configure_docking_layout(
        show_toolbar=self.config.show_toolbar,
        show_status_bar=self.config.show_status_bar,
    )

    io = imgui.GetIO()
    if io.WantSaveIniSettings:
      ux.save_settings(self.theme)
      io.WantSaveIniSettings = False

    with self._send_if_changes():
      # -- Main menu bar ------------------------------------------------------
      if imgui.BeginMainMenuBar():
        if imgui.BeginMenu('File'):
          if imgui.MenuItem('Quit'):
            self.close()
          imgui.EndMenu()
        if imgui.BeginMenu('Simulation'):
          imgui.EndMenu()
        if imgui.BeginMenu('View'):
          if imgui.MenuItem('Save Config'):
            ux.save_settings(self.theme)
          if imgui.MenuItem('Reset Config'):
            ux.reset_config()
            self.theme = ux.load_settings(def_theme=ux.GuiTheme.LIGHT)
            ux.setup_theme(self.theme)
          imgui.Separator()
          if imgui.MenuItem('Options', 'Tab', self.config.show_options):
            self.config.show_options = not self.config.show_options
          if imgui.MenuItem(
              'Inspector', 'Shift+Tab', self.config.show_inspector
          ):
            self.config.show_inspector = not self.config.show_inspector
          if imgui.MenuItem('Toolbar', '', self.config.show_toolbar):
            self.config.show_toolbar = not self.config.show_toolbar
          if imgui.MenuItem('View Gizmo', '', self.config.show_view_gizmo):
            self.config.show_view_gizmo = not self.config.show_view_gizmo
          if imgui.MenuItem('Status Bar', '', self.config.show_status_bar):
            self.config.show_status_bar = not self.config.show_status_bar
          imgui.Separator()
          if imgui.MenuItem('Info', 'F2', self.config.show_info):
            self.config.show_info = not self.config.show_info
          if imgui.MenuItem('Profiler', 'F3', self.config.show_profiler):
            self.config.show_profiler = not self.config.show_profiler
          if imgui.MenuItem('Console', '', self.config.show_console):
            self.config.show_console = not self.config.show_console
          imgui.Separator()
          changed, self.theme = ux.theme_menu_gui(self.theme)
          if changed:
            ux.setup_theme(self.theme)
            ux.save_settings(self.theme)
          imgui.EndMenu()
        if imgui.BeginMenu('Help'):
          version = f'Version {mujoco.mj_versionString()}'
          imgui.MenuItem(version)
          imgui.EndMenu()
        imgui.EndMainMenuBar()

      # -- Tool Bar -----------------------------------------------------------
      if self.config.show_toolbar:
        if imgui.Begin('ToolBar'):
          imgui.PushStyleVar(imgui.StyleVar.CellPadding, imgui.Vec2(0, 0))
          if imgui.BeginTable('##ToolBarTable', 2):
            imgui.TableSetupColumn('', int(imgui.TableColumnFlags.WidthStretch))
            imgui.TableSetupColumn('', int(imgui.TableColumnFlags.WidthFixed))

            imgui.TableNextColumn()
            self.reset_physics_gui()

            imgui.SameLine()
            ux.step_control_gui(self.step_control_state, self.ux_state)

            imgui.TableNextColumn()
            ux.camera_selection_gui(
                self.model,
                self.data,
                self.viewer.camera,
                self.ux_state,
            )

            imgui.SameLine()
            ux.label_selection_gui(self.viewer.vis_options)

            imgui.SameLine()
            ux.frame_selection_gui(self.viewer.vis_options)

            imgui.EndTable()
          imgui.PopStyleVar()
        imgui.End()

      # -- View gizmo ---------------------------------------------------------
      if self.config.show_view_gizmo:
        ux.view_gizmo_gui(
            self.model,
            self.data,
            self.viewer.camera,
            self.ux_state,
            workspace_rect,
        )

      # -- Left pane: Options -------------------------------------------------
      node_flags = int(imgui.TreeNodeFlags.SpanAvailWidth) | int(
          imgui.TreeNodeFlags.Framed
      )

      if self.config.show_options:
        _, self.config.show_options = imgui.Begin(
            'Options', self.config.show_options
        )
        ux.simulation_gui(
            self.model,
            self.data,
            self.step_control_state,
            self.ux_state,
            self.reset_physics,
            self.reload_model,
            self.align_camera,
            self.save_keyframe,
        )
        if imgui.TreeNodeEx('Physics Settings', node_flags):
          ux.physics_gui(self.model)
          imgui.TreePop()
        if imgui.TreeNodeEx('Rendering Settings', node_flags):
          ux.rendering_gui(
              self.model,
              self.viewer.vis_options,
              self.viewer.render_flags,
          )
          imgui.TreePop()
        if imgui.TreeNodeEx('Visibility Groups', node_flags):
          ux.groups_gui(self.model, self.viewer.vis_options)
          imgui.TreePop()
        if imgui.TreeNodeEx('Visualization', node_flags):
          ux.visualization_gui(
              self.model,
              self.viewer.vis_options,
              self.viewer.camera,
          )
          imgui.TreePop()
        imgui.End()

      # -- Right pane: Inspector ----------------------------------------------
      if self.config.show_inspector:
        _, self.config.show_inspector = imgui.Begin(
            'Inspector', self.config.show_inspector
        )
        if imgui.TreeNodeEx('Noise', node_flags):
          ux.noise_gui(self.step_control_state)
          imgui.TreePop()
        if imgui.TreeNodeEx('Joints', node_flags):
          with self.viewer.send_if_changes(
              lambda: messages.StateEvent.from_data(
                  self.model, self.data, int(mujoco.mjtState.mjSTATE_QPOS)
              ),
          ):
            ux.joints_gui(self.model, self.data, self.viewer.vis_options)
          imgui.TreePop()
        if imgui.TreeNodeEx('Controls', node_flags):
          with self.viewer.send_if_changes(
              lambda: messages.StateEvent.from_data(
                  self.model, self.data, int(mujoco.mjtState.mjSTATE_CTRL)
              ),
          ):
            ux.controls_gui(self.model, self.data, self.viewer.vis_options)
          imgui.TreePop()
        if imgui.TreeNodeEx(
            'Sensors', node_flags | int(imgui.TreeNodeFlags.DefaultOpen)
        ):
          ux.sensor_gui(self.model, self.data)
          imgui.TreePop()
        if imgui.TreeNodeEx('Watch', node_flags):
          ux.watch_gui(self.model, self.data, self.ux_state)
          imgui.TreePop()
        if imgui.TreeNodeEx('State', node_flags):
          state_sig = self.ux_state.state_sig
          with self.viewer.send_if_changes(
              lambda: messages.StateEvent.from_data(
                  self.model, self.data, state_sig
              ),
          ):
            ux.state_gui(self.model, self.data, self.ux_state)
          imgui.TreePop()
        imgui.End()

      self._forward_single_step()

      # -- Floating windows ---------------------------------------------------
      if self.config.show_profiler:
        _, self.config.show_profiler = imgui.Begin(
            'Profiler', self.config.show_profiler
        )
        ux.counts_gui(self.model, self.data)
        ux.convergence_gui(self.model, self.data)
        imgui.End()

      if self.config.show_console:
        expanded, self.config.show_console = imgui.Begin(
            'Console', self.config.show_console
        )
        if expanded:
          ux.console_gui()
        elif imgui.IsWindowCollapsed():
          imgui.SetWindowCollapsed(False)
          self.config.show_console = False
        imgui.End()

      if self.config.show_info:
        _, self.config.show_info = imgui.Begin('Info', self.config.show_info)
        paused = (
            self.step_control_state.get_pause_state() != sim.PauseState.UNPAUSED
        )
        ux.info_gui(self.model, self.data, paused, 0.0)
        imgui.End()

      # -- Status bar ---------------------------------------------------------
      if self.config.show_status_bar:
        imgui.PushStyleVar(imgui.StyleVar.CellPadding, imgui.Vec2(0, 0))
        imgui.PushStyleVar(imgui.StyleVar.FramePadding, imgui.Vec2(0, 0))
        imgui.PushStyleVar(imgui.StyleVar.WindowPadding, imgui.Vec2(0, 0))
        if imgui.Begin('StatusBar'):
          if imgui.BeginTable('##StatusBarTable', 2):
            imgui.TableSetupColumn('', int(imgui.TableColumnFlags.WidthStretch))
            imgui.TableSetupColumn('', int(imgui.TableColumnFlags.WidthFixed))
            imgui.TableNextColumn()
            imgui.TextUnformatted(self.status)
            imgui.TableNextColumn()
            self.config.show_console = ux.console_status_gui(
                self.config.show_console
            )
            imgui.EndTable()
        imgui.End()
        imgui.PopStyleVar(3)

    # Mirrored for viewers whose client moves the camera (client_camera), after
    # everything above that can switch the camera mode.
    self.viewer.camera_index = self.ux_state.camera_index

  @messages.handler(priority=messages.Priority.INTERNAL)
  def _on_step_control(self, event: messages.StepControlSnapshot) -> None:
    """Mirrors the sim's step control settings into ``step_control_state``."""
    self.step_control_state.set_pause_state(event.pause_state)
    self.step_control_state.set_speed(event.speed)
    self.step_control_state.set_noise_parameters(
        event.noise_scale, event.noise_rate
    )

  @messages.handler(priority=messages.Priority.INTERNAL)
  def _on_pause(self, event: messages.RequestPauseEvent) -> None:
    """Updates the viewer's pause state when requested by the sim."""
    self.step_control_state.set_pause_state(event.pause_state)

  @messages.handler(priority=messages.Priority.INTERNAL)
  def _on_history(self, event: sim_history.SimHistorySnapshot) -> None:
    """Mirrors the sim's history metadata into ``ux_state`` for the scrubber."""
    if self._viewer is None or self.model is None or self.data is None:
      return
    self.ux_state.history_size = event.size
    self.ux_state.history_index = event.index
    self.ux_state.sim_head_time = event.head_time

  @messages.handler(priority=messages.Priority.INTERNAL)
  def _on_post_model(self, event: messages.PostModelEvent) -> None:
    same_scene = event.path == self._model_path
    self._model_path = event.path
    if same_scene:
      self._reload_app_state(event.model)
    else:
      self._reset_app_state()

  @messages.handler(priority=messages.Priority.INTERNAL)
  def _on_exit(self, _: messages.ExitEvent) -> None:
    ux.save_settings(self.theme)

  @messages.handler(priority=messages.Priority.INTERNAL)
  def _on_theme(self, event: plugin.ThemeEvent) -> None:
    self.theme = event.theme
    ux.setup_theme(self.theme)
    ux.save_settings(self.theme)

  @messages.handler(priority=messages.Priority.INTERNAL)
  def _on_update(self, _: messages.UpdateEvent) -> None:
    self.update()

  @messages.handler(priority=messages.Priority.CRITICAL)
  def _on_build_gui(self, _: messages.BuildGuiEvent) -> None:
    self.build_gui()
