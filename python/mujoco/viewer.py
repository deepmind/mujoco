# Copyright 2022 DeepMind Technologies Limited
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Interactive GUI viewer for MuJoCo."""

import abc
import argparse
import atexit
import contextlib
import math
import os
import queue
import sys
import threading
import time
from typing import Callable, List, Optional, Tuple, Union
import weakref

import mujoco
from mujoco import _simulate
import numpy as np

try:
  import glfw
except ImportError as _err:
  raise ImportError(
      "mujoco.viewer requires 'glfw' to be installed."
  ) from _err
if not glfw._glfw:  # pylint: disable=protected-access
  raise RuntimeError('GLFW dynamic library handle is not available')
else:
  _simulate.set_glfw_dlhandle(glfw._glfw._handle)  # pylint: disable=protected-access

# Logarithmically spaced realtime slow-down coefficients (percent).
PERCENT_REALTIME = (
    100, 80, 66, 50, 40, 33, 25, 20, 16, 13,
    10, 8, 6.6, 5, 4, 3.3, 2.5, 2, 1.6, 1.3,
    1, 0.8, 0.66, 0.5, 0.4, 0.33, 0.25, 0.2, 0.16, 0.13,
    0.1
)  # fmt: skip

# Maximum time mis-alignment before re-sync.
MAX_SYNC_MISALIGN = 0.1

# Fraction of refresh available for simulation.
SIM_REFRESH_FRACTION = 0.7

CallbackType = Callable[[mujoco.MjModel, mujoco.MjData], None]
LoaderType = Callable[[], Tuple[mujoco.MjModel, mujoco.MjData]]
KeyCallbackType = Callable[[int], None]

# Loader function that also returns a file path for the GUI to display.
_LoaderWithPathType = Callable[[], Tuple[mujoco.MjModel, mujoco.MjData, str]]
_InternalLoaderType = Union[LoaderType, _LoaderWithPathType]

_Simulate = _simulate.Simulate


class Handle:
  """A handle for interacting with a MuJoCo viewer."""

  def __init__(
      self,
      sim: _Simulate,
      cam: mujoco.MjvCamera,
      opt: mujoco.MjvOption,
      pert: mujoco.MjvPerturb,
      user_scn: Optional[mujoco.MjvScene],
  ):
    self._sim = weakref.ref(sim)
    self._cam = cam
    self._opt = opt
    self._pert = pert
    self._user_scn = user_scn

  @property
  def cam(self):
    return self._cam

  @property
  def opt(self):
    return self._opt

  @property
  def perturb(self):
    return self._pert

  @property
  def user_scn(self):
    return self._user_scn

  @property
  def m(self):
    sim = self._sim()
    if sim is not None:
      return sim.m
    return None

  @property
  def d(self):
    sim = self._sim()
    if sim is not None:
      return sim.d
    return None

  @property
  def viewport(self):
    sim = self._sim()
    if sim is not None:
      return sim.viewport
    return None

  def set_figures(
      self, viewports_figures: Union[Tuple[mujoco.MjrRect, mujoco.MjvFigure],
                                   List[Tuple[mujoco.MjrRect, mujoco.MjvFigure]]]
  ):
    """Overlay figures on the viewer.

    Args:
      viewports_figures: Single tuple or list of tuples of (viewport, figure)
        viewport: Rectangle defining position and size of the figure
        figure: MjvFigure object containing the figure data to display
    """
    sim = self._sim()
    if sim is not None:
      # Convert single tuple to list if needed
      if isinstance(viewports_figures, tuple):
        viewports_figures = [viewports_figures]
      sim.set_figures(viewports_figures)

  def clear_figures(self):
    sim = self._sim()
    if sim is not None:
      sim.clear_figures()

  def set_texts(self, texts: Union[Tuple[Optional[int], Optional[int], Optional[str], Optional[str]],
                                            List[Tuple[Optional[int], Optional[int], Optional[str], Optional[str]]]]):
    """Overlay text on the viewer.

    Args:
      texts: Single tuple or list of tuples of (font, gridpos, text1, text2)
        font: Font style from mujoco.mjtFontScale
        gridpos: Position of text box from mujoco.mjtGridPos
        text1: Left text column, defaults to empty string if None
        text2: Right text column, defaults to empty string if None
    """
    sim = self._sim()
    if sim is not None:
      # Convert single tuple to list if needed
      if isinstance(texts, tuple):
        texts = [texts]

      # Convert None values to empty strings
      default_font = mujoco.mjtFontScale.mjFONTSCALE_150
      default_gridpos = mujoco.mjtGridPos.mjGRID_TOPLEFT
      processed_texts = [(
                        default_font if font is None else font,
                        default_gridpos if gridpos is None else gridpos,
                         "" if text1 is None else text1,
                         "" if text2 is None else text2)
                        for font, gridpos, text1, text2 in texts]

      sim.set_texts(processed_texts)

  def clear_texts(self):
    sim = self._sim()
    if sim is not None:
      sim.clear_texts()

  def set_images(
      self, viewports_images: Union[Tuple[mujoco.MjrRect, np.ndarray],
                                  List[Tuple[mujoco.MjrRect, np.ndarray]]]
  ):
    """Overlay images on the viewer.

    Args:
      viewports_images: Single tuple or list of tuples of (viewport, image)
        viewport: Rectangle defining position and size of the image
        image: RGB image with shape (height, width, 3)
    """
    sim = self._sim()
    if sim is not None:
      # Convert single tuple to list if needed
      if isinstance(viewports_images, tuple):
        viewports_images = [viewports_images]

      processed_images = []
      for viewport, image in viewports_images:
        targ_shape = (viewport.height, viewport.width)
        # Check if image is already the correct shape
        if image.shape[:2] != targ_shape:
          raise ValueError(f"Image shape {image.shape[:2]} does not match target shape {targ_shape}")
        flipped = np.flip(image, axis=0)
        contiguous = np.ascontiguousarray(flipped)
        processed_images.append((viewport, contiguous))
      sim.set_images(processed_images)

  def clear_images(self):
    sim = self._sim()
    if sim is not None:
      sim.clear_images()

  def close(self):
    sim = self._sim()
    if sim is not None:
      sim.exit()

  def _get_sim(self) -> Optional[_Simulate]:
    sim = self._sim()
    if sim is not None:
      try:
        return sim if sim.exitrequest == 0 else None
      except mujoco.UnexpectedError:
        # UnexpectedError is raised when accessing `exitrequest` after the
        # underlying simulate instance has been deleted in C++.
        return None
    return None

  def is_running(self) -> bool:
    return self._get_sim() is not None

  def lock(self):
    sim = self._get_sim()
    if sim is not None:
      return sim.lock()
    return contextlib.nullcontext()

  def sync(self, state_only: bool = False):
    sim = self._get_sim()
    if sim is not None:
      sim.sync(state_only)  # locks internally

  def update_hfield(self, hfieldid: int):
    sim = self._get_sim()
    if sim is not None:
      sim.update_hfield(hfieldid)  # locks internally and blocks until done

  def update_mesh(self, meshid: int):
    sim = self._get_sim()
    if sim is not None:
      sim.update_mesh(meshid)  # locks internally and blocks until done

  def update_texture(self, texid: int):
    sim = self._get_sim()
    if sim is not None:
      sim.update_texture(texid)  # locks internally and blocks until done

  def __enter__(self):
    return self

  def __exit__(self, exc_type, exc_val, exc_tb):
    self.close()


# Abstract base dispatcher class for systems that require UI calls to be made
# on a specific thread (e.g. macOS). This is subclassed by system-specific
# Python launcher (mjpython) to implement the required dispatching mechanism.
class _MjPythonBase(metaclass=abc.ABCMeta):

  def launch_on_ui_thread(
      self,
      model: mujoco.MjModel,
      data: mujoco.MjData,
      handle_return: Optional['queue.Queue[Handle]'],
      key_callback: Optional[KeyCallbackType],
  ):
    pass


# When running under mjpython, the launcher initializes this object.
_MJPYTHON: Optional[_MjPythonBase] = None


def _file_loader(path: str) -> _LoaderWithPathType:
  """Loads an MJCF model from file path."""

  def load(path=path) -> Tuple[mujoco.MjModel, mujoco.MjData, str]:
    if len(path) >= 4 and path[-4:] == '.mjb':
      m = mujoco.MjModel.from_binary_path(path)
    else:
      m = mujoco.MjModel.from_xml_path(path)
    d = mujoco.MjData(m)
    return m, d, path

  return load


def _reload(
    simulate: _Simulate,
    loader: _InternalLoaderType,
    notify_loaded: Optional[Callable[[], None]] = None,
) -> Optional[Tuple[mujoco.MjModel, mujoco.MjData]]:
  """Internal function for reloading a model in the viewer."""
  try:
    simulate.load_message('')  # path is unknown at this point
    load_tuple = loader()
  except Exception as e:  # pylint: disable=broad-except
    simulate.load_error = str(e)
    simulate.load_message_clear()
  else:
    m, d = load_tuple[:2]

    # If the loader does not raise an exception then we assume that it
    # successfully created mjModel and mjData. This is specified in the type
    # annotation, but we perform a runtime assertion here as well to prevent
    # possible segmentation faults.
    assert m is not None and d is not None

    path = load_tuple[2] if len(load_tuple) == 3 else ''
    simulate.load(m, d, path)

    # Make sure any load_error message is cleared
    simulate.load_error = ''

    if notify_loaded:
      notify_loaded()

    return m, d


def _physics_loop(simulate: _Simulate, loader: Optional[_InternalLoaderType]):
  """Physics loop for the GUI, to be run in a separate thread."""
  m: mujoco.MjModel = None  # pyrefly: ignore[bad-assignment]
  d: mujoco.MjData = None  # pyrefly: ignore[bad-assignment]
  ctrl_noise = np.array([])
  reload = True

  # CPU-sim synchronization point.
  synccpu = 0.0
  syncsim = 0.0

  # Run until asked to exit.
  while not simulate.exitrequest:
    if simulate.droploadrequest:
      simulate.droploadrequest = 0
      loader = _file_loader(simulate.dropfilename)
      reload = True

    if simulate.uiloadrequest:
      simulate.uiloadrequest_decrement()
      reload = True

    if reload and loader is not None:
      result = _reload(simulate, loader)
      if result is not None:
        m, d = result
        ctrl_noise = np.zeros((m.nu,))

    reload = False

    # Sleep for 1 ms or yield, to let main thread run.
    if simulate.run != 0 and simulate.busywait != 0:
      time.sleep(0)
    else:
      time.sleep(0.001)

    with simulate.lock():
      if m is not None:
        assert d is not None
        if simulate.run:
          stepped = False
          # Record CPU time at start of iteration.
          startcpu = time.time()

          elapsedcpu = startcpu - synccpu
          elapsedsim = d.time - syncsim

          # Inject noise.
          if simulate.ctrl_noise_std != 0.0:
            # Convert rate and scale to discrete time (Ornstein–Uhlenbeck).
            rate = math.exp(
                -m.opt.timestep / max(simulate.ctrl_noise_rate, mujoco.mjMINVAL)
            )
            scale = simulate.ctrl_noise_std * math.sqrt(1 - rate * rate)

            for i in range(m.nu):
              # Update noise.
              ctrl_noise[i] = rate * ctrl_noise[
                  i
              ] + scale * mujoco.mju_standardNormal(None)

              # Apply noise.
              d.ctrl[i] = ctrl_noise[i]

          # Requested slow-down factor.
          slowdown = 100 / PERCENT_REALTIME[simulate.real_time_index]

          # Misalignment: distance from target sim time > MAX_SYNC_MISALIGN.
          misaligned = (
              abs(elapsedcpu / slowdown - elapsedsim) > MAX_SYNC_MISALIGN
          )

          # Out-of-sync (for any reason): reset sync times, step.
          if (
              elapsedsim < 0
              or elapsedcpu < 0
              or synccpu == 0
              or misaligned
              or simulate.speed_changed
          ):
            # Re-sync.
            synccpu = startcpu
            syncsim = d.time
            simulate.speed_changed = False

            # Run single step, let next iteration deal with timing.
            mujoco.mj_step(m, d)
            stepped = True

          # In-sync: step until ahead of cpu.
          else:
            measured = False
            prevsim = d.time
            refreshtime = SIM_REFRESH_FRACTION / simulate.refresh_rate
            # Step while sim lags behind CPU and within refreshtime.
            while (
                (d.time - syncsim) * slowdown < (time.time() - synccpu)
            ) and ((time.time() - startcpu) < refreshtime):
              # Measure slowdown before first step.
              if not measured and elapsedsim:
                simulate.measured_slowdown = elapsedcpu / elapsedsim
                measured = True

              # Call mj_step.
              mujoco.mj_step(m, d)
              stepped = True

              # Break if reset.
              if d.time < prevsim:
                break

          # save current state to history buffer
          if stepped:
            simulate.add_to_history()

        else:  # simulate.run is False: GUI is paused.

          # Run mj_forward, to update rendering and joint sliders.
          mujoco.mj_forward(m, d)
          simulate.speed_changed = True


def _launch_internal(
    model: Optional[mujoco.MjModel] = None,
    data: Optional[mujoco.MjData] = None,
    *,
    run_physics_thread: bool,
    loader: Optional[_InternalLoaderType] = None,
    handle_return: Optional['queue.Queue[Handle]'] = None,
    key_callback: Optional[KeyCallbackType] = None,
    show_left_ui: bool = True,
    show_right_ui: bool = True,
) -> None:
  """Internal API, so that the public API has more readable type annotations."""
  if model is None and data is not None:
    raise ValueError('mjData is specified but mjModel is not')
  elif callable(model) and data is not None:
    raise ValueError(
        'mjData should not be specified when an mjModel loader is used'
    )
  elif loader is not None and model is not None:
    raise ValueError('model and loader are both specified')
  elif run_physics_thread and handle_return is not None:
    raise ValueError('run_physics_thread and handle_return are both specified')

  if loader is None and model is not None:

    def _loader(m=model, d=data) -> Tuple[mujoco.MjModel, mujoco.MjData]:
      if d is None:
        d = mujoco.MjData(m)
      return m, d

    loader = _loader

  cam = mujoco.MjvCamera()
  opt = mujoco.MjvOption()
  pert = mujoco.MjvPerturb()
  if model and not run_physics_thread:
    user_scn = mujoco.MjvScene(model, _Simulate.MAX_GEOM)
  else:
    user_scn = None
  simulate = _Simulate(
      cam, opt, pert, user_scn, run_physics_thread, key_callback
  )

  simulate.ui0_enable = show_left_ui
  simulate.ui1_enable = show_right_ui

  # Initialize GLFW if not using mjpython.
  if _MJPYTHON is None:
    if not glfw.init():
      raise mujoco.FatalError('could not initialize GLFW')
    atexit.register(glfw.terminate)

  notify_loaded = None
  if handle_return:
    notify_loaded = lambda: handle_return.put_nowait(
        Handle(simulate, cam, opt, pert, user_scn)
    )

  if run_physics_thread:
    side_thread = threading.Thread(
        target=_physics_loop, args=(simulate, loader)
    )
  else:
    side_thread = threading.Thread(
        target=_reload, args=(simulate, loader, notify_loaded)
    )

  def make_exit(simulate):
    def exit_simulate():
      simulate.exit()

    return exit_simulate

  exit_simulate = make_exit(simulate)
  atexit.register(exit_simulate)

  side_thread.start()
  simulate.render_loop()
  atexit.unregister(exit_simulate)
  side_thread.join()
  simulate.destroy()


def launch(
    model: Optional[mujoco.MjModel] = None,
    data: Optional[mujoco.MjData] = None,
    *,
    loader: Optional[LoaderType] = None,
    show_left_ui: bool = True,
    show_right_ui: bool = True,
) -> None:
  """Launches the Simulate GUI."""
  _launch_internal(
      model,
      data,
      run_physics_thread=True,
      loader=loader,
      show_left_ui=show_left_ui,
      show_right_ui=show_right_ui,
  )


def launch_from_path(path: str) -> None:
  """Launches the Simulate GUI from file path."""
  _launch_internal(run_physics_thread=True, loader=_file_loader(path))


def launch_passive(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    *,
    key_callback: Optional[KeyCallbackType] = None,
    show_left_ui: bool = True,
    show_right_ui: bool = True,
) -> Handle:
  """Launches a passive Simulate GUI without blocking the running thread."""
  if not isinstance(model, mujoco.MjModel):
    raise ValueError(f'`model` is not a mujoco.MjModel: got {model!r}')
  if not isinstance(data, mujoco.MjData):
    raise ValueError(f'`data` is not a mujoco.MjData: got {data!r}')
  if key_callback is not None and not callable(key_callback):
    raise ValueError(f'`key_callback` is not callable: got {key_callback!r}')

  mujoco.mj_forward(model, data)
  handle_return = queue.Queue(1)

  if sys.platform != 'darwin':
    thread = threading.Thread(
        target=_launch_internal,
        args=(model, data),
        kwargs=dict(
            run_physics_thread=False,
            handle_return=handle_return,
            key_callback=key_callback,
            show_left_ui=show_left_ui,
            show_right_ui=show_right_ui,
        ),
    )
    thread.daemon = True
    thread.start()
  else:
    if not isinstance(_MJPYTHON, _MjPythonBase):
      raise RuntimeError(
          '`launch_passive` requires that the Python script be run under '
          '`mjpython` on macOS'
      )
    _MJPYTHON.launch_on_ui_thread(
        model,
        data,
        handle_return,
        key_callback,
        show_left_ui,
        show_right_ui,
    )

  return handle_return.get()


if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  parser.add_argument('--mjcf', type=str, help='Path to MJCF file.')
  args, _ = parser.parse_known_args()
  if args.mjcf is not None:
    launch_from_path(os.path.expanduser(args.mjcf))
  else:
    launch()
