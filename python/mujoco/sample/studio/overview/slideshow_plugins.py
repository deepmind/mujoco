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
r"""Interactive slideshow plugins for MuJoCo 4.0 features.

Standalone web viewer::

  python -m mujoco.sample.studio.overview.slideshow

Inside the Studio viewer::

  python -m mujoco.studio.viewer \
      --gfx=web \
      --plugins=python/mujoco/sample/studio/overview/slideshow_plugins.py
"""

from collections.abc import Callable, Sequence
import dataclasses
import enum
import logging
import math
import os
import pathlib
import queue
import threading
import time

import mujoco
from mujoco.studio import app as studio_app
from mujoco.studio import messages
from mujoco.studio import parser
from mujoco.studio import plugin
from mujoco.studio import sim
from mujoco.studio import sim_history
from mujoco.studio import ux
from mujoco.studio import viewer_utils
from mujoco.studio.app import inputs
import numpy as np

from mujoco.studio.dear_imgui import dear_imgui as imgui

logger = logging.getLogger(__name__)


# Icons are drawn with the ImGui draw list (see ``_draw_icon``) rather than
# with font glyphs, so they sit exactly centered in their buttons.
ICON_PREV = 'prev'  # chevron-left
ICON_NEXT = 'next'  # chevron-right
ICON_PLAY = 'play'
ICON_PAUSE = 'pause'
ICON_RESET = 'reset'  # circular arrow
ICON_HELP = 'help'  # circled question mark
ICON_COLLAPSE = 'collapse'  # chevron-down
ICON_EXPAND = 'expand'  # chevron-up


class CardState(enum.Enum):
  """Display state of the floating slideshow overlay card."""

  EXPANDED = 'expanded'
  COMPACT = 'compact'
  HIDDEN = 'hidden'


def _hex(h: str, a: float = 1.0) -> imgui.Vec4:
  """Converts a ``#RRGGBB`` hex color and alpha into an ``imgui.Vec4``."""
  clean = h.lstrip('#')
  r = int(clean[0:2], 16) / 255.0
  g = int(clean[2:4], 16) / 255.0
  b = int(clean[4:6], 16) / 255.0
  return imgui.Vec4(r, g, b, a)


def _mix(a: imgui.Vec4, b: imgui.Vec4, t: float) -> imgui.Vec4:
  """Linearly interpolates two ``imgui.Vec4`` colors by ``t`` in ``[0, 1]``."""
  return imgui.Vec4(
      a.x + (b.x - a.x) * t,
      a.y + (b.y - a.y) * t,
      a.z + (b.z - a.z) * t,
      a.w + (b.w - a.w) * t,
  )


def _over(base: imgui.Vec4, top: imgui.Vec4) -> imgui.Vec4:
  """Composites ``top`` over ``base`` by ``top``'s alpha, keeping base alpha."""
  mixed = _mix(base, top, top.w)
  return imgui.Vec4(mixed.x, mixed.y, mixed.z, base.w)


_CLEAR = imgui.Vec4(0.0, 0.0, 0.0, 0.0)

# The Gemini sparkle gradient, stepped across the slides for the active dot
# and the slide counter.
_GEMINI_RAMP: tuple[imgui.Vec4, ...] = (
    _hex('#4285F4'),
    _hex('#9B72CB'),
    _hex('#D96570'),
)


@dataclasses.dataclass(frozen=True)
class OverlayPalette:
  """Color tokens for the floating slideshow card, buttons, and popups."""

  surface: imgui.Vec4
  surface_hi: imgui.Vec4
  container: imgui.Vec4
  on_container: imgui.Vec4
  outline: imgui.Vec4
  on_surface: imgui.Vec4
  on_surface_var: imgui.Vec4
  muted: imgui.Vec4
  accent: imgui.Vec4
  accent_text: imgui.Vec4
  accent_soft: imgui.Vec4
  hover: imgui.Vec4
  press: imgui.Vec4
  selected: imgui.Vec4
  dot_idle: imgui.Vec4
  tooltip_bg: imgui.Vec4
  tooltip_text: imgui.Vec4
  dot_ramp: tuple[imgui.Vec4, ...] = ()


@dataclasses.dataclass(frozen=True)
class OverlayShape:
  """Geometry, typography, and layout metrics of the card."""

  card_rounding: float
  btn_rounding: float
  btn_size: float
  pad_x: float
  pad_y: float
  title_px: float
  caption_px: float
  eyebrow_px: float
  icon_px: float
  dot_w: float
  dot_px: float
  dot_px_active: float
  wrap_width: float
  row_gap: float
  bottom_margin: float


@dataclasses.dataclass(frozen=True)
class ThemeSpec:
  """The card's look for one of the Studio GUI themes."""

  key: str
  label: str
  shape: OverlayShape
  palette: OverlayPalette


# Material 3 style: pill-rounded card and circular icon buttons.
_SHAPE = OverlayShape(
    card_rounding=22.0,
    btn_rounding=16.0,
    btn_size=32.0,
    pad_x=20.0,
    pad_y=16.0,
    title_px=20.0,
    caption_px=15.0,
    eyebrow_px=12.0,
    icon_px=18.0,
    dot_w=16.0,
    dot_px=10.0,
    dot_px_active=14.0,
    wrap_width=520.0,
    row_gap=12.0,
    bottom_margin=24.0,
)

_GEMINI_DARK = ThemeSpec(
    key='gemini_dark',
    label='Gemini Dark',
    shape=_SHAPE,
    palette=OverlayPalette(
        surface=_hex('#131314', 0.90),
        surface_hi=_hex('#1E1F20', 0.98),
        container=_hex('#0842A0'),
        on_container=_hex('#D3E3FD'),
        outline=_hex('#444746'),
        on_surface=_hex('#E3E3E3'),
        on_surface_var=_hex('#C4C7C5'),
        muted=_hex('#8E918F'),
        accent=_hex('#8AB4F8'),
        accent_text=_hex('#8AB4F8'),
        accent_soft=_hex('#8AB4F8', 0.14),
        hover=_hex('#E3E3E3', 0.08),
        press=_hex('#E3E3E3', 0.12),
        selected=_hex('#8AB4F8', 0.16),
        dot_idle=_hex('#5E5E5E'),
        tooltip_bg=_hex('#1E1F20', 0.98),
        tooltip_text=_hex('#E3E3E3'),
        dot_ramp=_GEMINI_RAMP,
    ),
)

_GEMINI_LIGHT = ThemeSpec(
    key='gemini_light',
    label='Gemini Light',
    shape=_SHAPE,
    palette=OverlayPalette(
        surface=_hex('#F8F9FA', 0.94),
        surface_hi=_hex('#FFFFFF', 0.98),
        container=_hex('#D3E3FD'),
        on_container=_hex('#041E49'),
        outline=_hex('#DADCE0'),
        on_surface=_hex('#1F1F1F'),
        on_surface_var=_hex('#444746'),
        muted=_hex('#5F6368'),
        accent=_hex('#1A73E8'),
        accent_text=_hex('#0B57D0'),
        accent_soft=_hex('#E8F0FE'),
        hover=_hex('#1F1F1F', 0.08),
        press=_hex('#1F1F1F', 0.12),
        selected=_hex('#1A73E8', 0.12),
        dot_idle=_hex('#C4C7C5'),
        tooltip_bg=_hex('#FFFFFF', 0.98),
        tooltip_text=_hex('#1F1F1F'),
        dot_ramp=_GEMINI_RAMP,
    ),
)

# The card follows the viewer's GUI theme (``plugin.ThemeEvent``): the light
# Studio theme gets the light card, the dark and classic themes the dark one.
THEMES: dict[ux.GuiTheme, ThemeSpec] = {
    ux.GuiTheme.LIGHT: _GEMINI_LIGHT,
    ux.GuiTheme.DARK: _GEMINI_DARK,
    ux.GuiTheme.CLASSIC: _GEMINI_DARK,
}
DEFAULT_THEME = ux.GuiTheme.DARK


@dataclasses.dataclass(frozen=True)
class CameraPreset:
  """Interactive free-camera framing for a slide."""

  lookat: tuple[float, float, float]
  distance: float
  azimuth: float
  elevation: float


@dataclasses.dataclass(frozen=True)
class SlideSpec:
  """Declarative specification for one MuJoCo 4.0 feature slide."""

  slug: str
  category: str
  title: str
  caption: str
  model_relpath: str = ''
  build_model_fn: Callable[[], mujoco.MjModel] | None = None
  init_fn: Callable[[mujoco.MjModel, mujoco.MjData], None] | None = None
  pre_step_fn: Callable[[mujoco.MjModel, mujoco.MjData], None] | None = None
  camera: CameraPreset | None = None
  vis_island: bool = False


@dataclasses.dataclass(frozen=True)
class SelectSlideEvent(messages.Event):
  """Viewer-to-sim request to advance by ``delta`` or jump to ``index``."""

  delta: int = 0
  index: int | None = None


@dataclasses.dataclass(frozen=True)
class SlideChangedEvent(messages.Event):
  """Sim-to-viewer notification sent whenever the active slide/model changes."""

  index: int
  total: int
  slug: str
  category: str
  title: str
  caption: str
  camera: CameraPreset | None = None
  vis_island: bool = False
  is_custom: bool = False


def _step_ipc_bag(model: mujoco.MjModel, data: mujoco.MjData) -> None:
  """Holds the drawstring at the top of its range so the mouth gathers shut."""
  if model.nu > 0:
    data.ctrl[0] = 1.0


def _init_fridge_door(model: mujoco.MjModel, data: mujoco.MjData) -> None:
  """Starts the door swinging gently inward so magnetic adhesion latches it."""
  if model.nq >= 1 and model.nv >= 1:
    data.qpos[0] = 0.30
    data.qvel[0] = -0.65


SLIDES: tuple[SlideSpec, ...] = (
    SlideSpec(
        slug='pbr',
        category='Rendering',
        title='Physically Based Rendering',
        caption=(
            'MuJoCo 4.0 renders with Google Filament on desktop and in the'
            ' browser. Scenes combine metallic-roughness materials,'
            ' image-based lighting, soft shadows, fog, and bloom.'
        ),
        model_relpath='welcome/welcome.xml',
    ),
    SlideSpec(
        slug='ccd',
        category='Speculative Contacts',
        title='Speculative Contacts: Baseball Bat',
        caption=(
            'Speculative contacts sweep fast-moving geometry across each step'
            ' to stop thin objects from tunneling through one another. Even at'
            ' a coarse 10 ms timestep, a thin bat cleanly hits a'
            ' fast-pitched baseball.'
        ),
        model_relpath='speculative/flipper.xml',
    ),
    SlideSpec(
        slug='chain',
        category='Speculative Contacts',
        title='Speculative Contacts: Falling Chain',
        caption=(
            'Four free-falling chain links built from thin 6 mm capsules drop'
            ' onto a fixed catch bar at a 6 ms timestep. Speculative contacts'
            ' catch every link-to-link impact even when links move more than a'
            ' capsule radius per step.'
        ),
        model_relpath='speculative/chain.xml',
    ),
    SlideSpec(
        slug='surfacevel',
        category='Contact & Actuation',
        title='Surface Velocity: Luggage Carousel',
        caption=(
            'Static geometry can apply tangential or rotational surface'
            ' velocity through friction without adding moving joints. Three'
            ' conveyor belts feed bags onto a spinning carousel using contact'
            ' alone.'
        ),
        model_relpath='surfacevel/carousel.xml',
        camera=CameraPreset(
            lookat=(-0.25, -1.55, 0.55),
            distance=5.6,
            azimuth=118.0,
            elevation=-26.0,
        ),
    ),
    SlideSpec(
        slug='adhesion',
        category='Contact & Actuation',
        title='Magnetic Adhesion',
        caption=(
            'Contacts can apply attractive normal forces to model magnets,'
            ' suction cups, and sticky surfaces in the solver. A swinging'
            ' fridge door snaps shut across a 3 mm gap, and pulling hard'
            ' enough breaks the seal.'
        ),
        model_relpath='adhesion/fridge_door.xml',
        init_fn=_init_fridge_door,
        camera=CameraPreset(
            lookat=(-0.08, 0.0, 0.42),
            distance=1.25,
            azimuth=-125.0,
            elevation=-18.0,
        ),
    ),
    SlideSpec(
        slug='cards',
        category='Contact & Integration',
        title='House of Cards',
        caption=(
            'Twenty-six playing cards stand four stories tall on friction'
            ' alone, held steady by elliptic friction cones and the noslip'
            ' solver. Drag a card with the mouse to test how much the'
            ' structure can take before toppling.'
        ),
        model_relpath='cards/house_of_cards.xml',
        camera=CameraPreset(
            lookat=(0.0, 0.0, 0.105),
            distance=0.46,
            azimuth=150.0,
            elevation=-15.0,
        ),
    ),
    SlideSpec(
        slug='arch',
        category='Contact & Integration',
        title='Dry-Stone Gothic Arch',
        caption=(
            'Unmortared stone wedges and a keystone hold together by friction'
            ' alone. Elliptic friction cones and the discrete integrator keep'
            ' the arch standing steady under gravity at an 8 ms timestep.'
        ),
        model_relpath='arch/gothic.xml',
        camera=CameraPreset(
            lookat=(0.0, 0.0, 0.95),
            distance=3.6,
            azimuth=152.0,
            elevation=-18.0,
        ),
    ),
    SlideSpec(
        slug='wire',
        category='Plasticity',
        title='Plastic Joints & Tendons',
        caption=(
            'Joint springs and tendons permanently shift their rest positions'
            ' once loaded past their yield threshold. Identical weights drop'
            ' onto steel, brass, and copper beams and cables: steel rebounds,'
            ' brass bends slightly, and copper stays deformed.'
        ),
        model_relpath='plastic/wire.xml',
        camera=CameraPreset(
            lookat=(0.34, 0.0, 0.34),
            distance=1.45,
            azimuth=122.0,
            elevation=-22.0,
        ),
    ),
    SlideSpec(
        slug='paper',
        category='Plasticity',
        title='Creased Paper',
        caption=(
            'Thin shells with inextensible edges and a bending yield threshold'
            ' crease permanently wherever they fold. Pulled through brass'
            ' rings, the elastic cloth on the left unfolds flat while the'
            ' paper on the right keeps its radial creases.'
        ),
        model_relpath='plastic/paper.xml',
    ),
    SlideSpec(
        slug='foil',
        category='Plasticity',
        title='Foil Hailstorm',
        caption=(
            'Shell plasticity uses separate yield thresholds for stretching'
            ' and bending, so impacts leave crisp dents without sagging.'
            ' Falling steel balls bounce off spring steel on the left while'
            ' permanently denting metallic foil on the right.'
        ),
        model_relpath='plastic/foil.xml',
    ),
    SlideSpec(
        slug='clay',
        category='Plasticity',
        title='Stamped Clay',
        caption=(
            'Volumetric bodies deform permanently once stress exceeds their'
            ' yield limit. A weighted press stamps a cross die into soft clay'
            ' and retracts, leaving a crisp impression instead of springing'
            ' back.'
        ),
        model_relpath='plastic/clay.xml',
        camera=CameraPreset(
            lookat=(0.0, 0.03, 0.10),
            distance=0.68,
            azimuth=135.0,
            elevation=-28.0,
        ),
    ),
    SlideSpec(
        slug='dipper',
        category='Deformables',
        title='Elastic Rods in Mechanisms',
        caption=(
            'Discrete elastic rods model beams and cables that bend and twist,'
            ' coupled to rigid bodies through ordinary joints and constraints.'
            ' Here a crank-driven piston rocks a flexible dipper arm carrying'
            ' a suspended weight.'
        ),
        model_relpath='flex/mechanisms/dipper.xml',
    ),
    SlideSpec(
        slug='ipc',
        category='Deformables',
        title='Penetration-Free Cloth Contact',
        caption=(
            "MuJoCo's IPC contact mode checks every step for intersections,"
            ' so cloth never passes through itself or the objects it touches.'
            ' Here a ribbon drawstring slides inside the hem of a bag hanging'
            ' from a hoop and cinches it shut by contact alone.'
        ),
        model_relpath='flex/bag_hoop.xml',
        pre_step_fn=_step_ipc_bag,
        camera=CameraPreset(
            lookat=(0.0, 0.0, 0.22),
            distance=0.68,
            azimuth=125.0,
            elevation=-22.0,
        ),
    ),
    SlideSpec(
        slug='sleep',
        category='Performance',
        title='Island Sleeping',
        caption=(
            'Island sleeping removes resting groups of bodies from the solver'
            ' until an impact wakes them. Ninety-six dominoes start asleep at'
            ' near-zero cost, waking row by row as the toppling wave arrives.'
        ),
        model_relpath='sleep/dominos.xml',
        vis_island=True,
        camera=CameraPreset(
            lookat=(0.65, 0.0, 0.10),
            distance=2.55,
            azimuth=122.0,
            elevation=-32.0,
        ),
    ),
)


def _scene_default_speed(model: mujoco.MjModel) -> float:
  """Returns ``model``'s authored speed in percent of real time."""
  # ``vis.global.realtime`` is float32 in C (e.g. 0.2f -> 0.200000003).
  pct = round(100.0 * float(model.vis.global_.realtime), 4)
  return min(100.0, max(0.1, pct))


def find_model_dir(explicit_dir: str | os.PathLike[str] = '') -> pathlib.Path:
  """Locates MuJoCo's top-level ``model/`` directory.

  An explicit directory or ``MUJOCO_MODEL_DIR`` overrides the default search,
  e.g. when running from an installed wheel outside a source checkout.

  Args:
    explicit_dir: A directory to use instead, or empty.

  Returns:
    The first candidate directory that holds ``welcome/welcome.xml``.

  Raises:
    FileNotFoundError: If no candidate directory holds ``welcome/welcome.xml``.
  """
  candidates: list[pathlib.Path] = []
  if explicit_dir:
    candidates.append(pathlib.Path(explicit_dir))
  env_dir = os.environ.get('MUJOCO_MODEL_DIR', '')
  if env_dir:
    candidates.append(pathlib.Path(env_dir))
  # Not resolve(): in a runfiles tree the module is a symlink into a content
  # store, and the models live relative to the symlink, not its target.
  here = pathlib.Path(__file__).absolute()
  if len(here.parents) > 5:
    candidates.append(here.parents[5] / 'model')

  for candidate in candidates:
    if (candidate / 'welcome' / 'welcome.xml').is_file():
      return candidate

  searched = ', '.join(str(c) for c in candidates)
  raise FileNotFoundError(
      'Could not locate MuJoCo model directory (welcome/welcome.xml). '
      f'Set MUJOCO_MODEL_DIR or pass --model_dir. Searched: {searched}'
  )


def resolve_slide_index(
    selector: str | int, slides: Sequence[SlideSpec] = SLIDES
) -> int:
  """Resolves a 1-based slide number or slug (e.g. ``'ipc'`` or ``'2'``)."""
  if not slides:
    raise ValueError('Slide deck is empty.')
  if isinstance(selector, int):
    return selector % len(slides)
  text = selector.strip().lower()
  if not text:
    return 0
  if text.isdigit():
    one_based = int(text)
    if not 1 <= one_based <= len(slides):
      raise ValueError(
          f'Slide number {one_based} out of range 1..{len(slides)}.'
      )
    return one_based - 1
  for i, slide in enumerate(slides):
    if slide.slug.lower() == text:
      return i
  slugs = ', '.join(s.slug for s in slides)
  raise ValueError(
      f'Unknown slide {selector!r}; choose 1..{len(slides)} or: {slugs}'
  )


def is_draggable_body(model: mujoco.MjModel, body: int) -> bool:
  """Returns whether a mouse drag can move ``body``.

  True for bodies that hang off a joint somewhere above them and for mocap
  bodies; false for the world (and so the floor), and for bodies welded to the
  world, which a drag should orbit the camera around instead.

  Args:
    model: The model.
    body: A body id, as returned by picking.
  """
  if body <= 0 or body >= model.nbody:
    return False
  return int(model.body_weldid[body]) != 0 or int(model.body_mocapid[body]) >= 0


def apply_interactive_camera(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    camera: mujoco.MjvCamera,
    preset: CameraPreset | None = None,
) -> None:
  """Configures an interactive ``mjCAMERA_FREE`` pose for a slide.

  Authored ``<camera>`` elements in MJCF use ``mjCAMERA_FIXED`` when selected
  directly, which locks out mouse orbit/pan/zoom. Converting the authored camera
  pose into ``mjCAMERA_FREE`` spherical coordinates preserves the exact initial
  framing on frame 1 while keeping full mouse camera controls active.

  Args:
    model: Loaded MuJoCo model.
    data: Forwarded MuJoCo data.
    camera: Viewer camera to configure in place.
    preset: Explicit spherical free-camera preset, or None to convert the
      model's ``visual/global/cameraid`` camera (falling back to
      ``mjv_defaultFreeCamera`` when it is unset, like MuJoCo Studio).
  """
  mujoco.mjv_defaultFreeCamera(model, camera)
  camera.type = int(mujoco.mjtCamera.mjCAMERA_FREE)
  camera.fixedcamid = -1
  camera.trackbodyid = -1

  if preset is not None:
    camera.lookat[:] = preset.lookat
    camera.distance = preset.distance
    camera.azimuth = preset.azimuth
    camera.elevation = preset.elevation
    return

  cam_id = model.vis.global_.cameraid
  if not 0 <= cam_id < model.ncam:
    return

  pos = np.asarray(data.cam_xpos[cam_id], dtype=np.float64)
  xmat = np.asarray(data.cam_xmat[cam_id], dtype=np.float64).reshape(3, 3)
  forward = -xmat[:, 2]
  norm = float(np.linalg.norm(forward))
  if norm < 1e-9:
    return
  forward = forward / norm

  center = np.asarray(model.stat.center, dtype=np.float64)
  proj_dist = float(np.dot(center - pos, forward))
  fallback_dist = max(0.5, 1.5 * float(model.stat.extent))
  distance = proj_dist if proj_dist > 0.35 else fallback_dist

  camera.lookat[:] = pos + distance * forward
  camera.distance = distance
  camera.azimuth = math.degrees(
      math.atan2(float(forward[1]), float(forward[0]))
  )
  camera.elevation = math.degrees(
      math.asin(max(-1.0, min(1.0, float(forward[2]))))
  )


def deck_compile_order(current: int, total: int) -> list[int]:
  """Returns the other slide indices, nearest to ``current`` first.

  Left and Right from the current slide are the likeliest next slides, so the
  deck compiles (and the browser prefetches) outwards from it: next, previous,
  next-but-one, and so on.

  Args:
    current: The index of the current slide.
    total: The number of slides.
  """
  order: list[int] = []
  for step in range(1, total):
    for candidate in ((current + step) % total, (current - step) % total):
      if candidate != current and candidate not in order:
        order.append(candidate)
  return order


class SlideshowSim(plugin.SimPlugin):
  """Sim-side plugin that loads MuJoCo 4.0 showcase scenes and steps scripts.

  With ``prefetch`` on (the default), the slides other than the first are
  compiled once on a background thread, nearest slides first, and kept for the
  rest of the session; each finished slide is announced to the viewer with a
  ``PreloadModelEvent`` so the web viewer's browsers download it ahead of time.
  Switching slides then neither compiles nor downloads anything.
  """

  _slides: Sequence[SlideSpec] = SLIDES
  _index: int = 0
  _autoplay_seconds: float = 0.0
  _model_dir: pathlib.Path | None = None
  _explicit_model_dir: str = ''
  _slide_loaded_wall_time: float = 0.0
  _loading_slide: bool = False
  _is_custom_model: bool = False
  _prefetch: bool = True
  _loaded_model_path: str = ''
  _compile_cancel: threading.Event | None = None

  def __init__(
      self,
      *,
      slides: Sequence[SlideSpec] = SLIDES,
      initial_slide: int = 0,
      autoplay_seconds: float = 0.0,
      model_dir: str | os.PathLike[str] = '',
      prefetch: bool = True,
  ) -> None:
    self._slides = tuple(slides)
    self._index = initial_slide % len(self._slides) if self._slides else 0
    self._autoplay_seconds = max(0.0, float(autoplay_seconds))
    self._explicit_model_dir = os.fspath(model_dir) if model_dir else ''
    self._model_dir = None
    self._slide_loaded_wall_time = time.monotonic()
    self._loading_slide = False
    self._is_custom_model = False
    self._prefetch = prefetch
    self._loaded_model_path = ''
    # Compiled slides by index, kept for the session (sim thread only), and
    # the queue the compile thread delivers them through.
    self._deck: dict[int, tuple[mujoco.MjModel, str]] = {}
    self._compiled: queue.Queue[tuple[int, mujoco.MjModel, str]] = queue.Queue()
    self._compile_thread: threading.Thread | None = None
    # Set to stop the compile thread before its next slide.
    self._compile_cancel: threading.Event | None = None

  @property
  def index(self) -> int:
    return self._index

  @property
  def current_slide(self) -> SlideSpec:
    return self._slides[self._index]

  def _get_model_dir(self) -> pathlib.Path:
    if self._model_dir is None:
      self._model_dir = find_model_dir(self._explicit_model_dir)
    return self._model_dir

  def compile_slide(self, index: int) -> tuple[mujoco.MjModel, str]:
    """Compiles/loads ``slides[index]`` and returns ``(model, model_path)``."""
    slide = self._slides[index % len(self._slides)]
    if slide.build_model_fn is not None:
      return slide.build_model_fn(), ''
    path = self._get_model_dir() / slide.model_relpath
    return parser.load(str(path)), str(path)

  def _start_deck_compile(self) -> None:
    """Compiles the other slides on a background thread, nearest first.

    MuJoCo releases the GIL while parsing and compiling, so this runs alongside
    the simulation; finished slides are picked up by ``_collect_compiled`` on
    the sim thread.
    """
    if not self._prefetch or self._compile_thread is not None:
      return
    pending = [
        i
        for i in deck_compile_order(self._index, len(self._slides))
        if i not in self._deck
    ]
    if not pending:
      return

    cancel = threading.Event()

    def run() -> None:
      for i in pending:
        if cancel.is_set():
          return
        try:
          model, path = self.compile_slide(i)
        except Exception:  # pylint: disable=broad-exception-caught
          logger.exception('Compiling slide %r failed.', self._slides[i].slug)
          continue
        if cancel.is_set():
          return
        self._compiled.put((i, model, path))

    self._compile_cancel = cancel
    self._compile_thread = threading.Thread(target=run, daemon=True)
    self._compile_thread.start()

  def _stop_deck_compile(self) -> None:
    """Cancels and joins the background deck compile thread, if running."""
    if self._compile_thread is not None:
      if self._compile_cancel is not None:
        self._compile_cancel.set()
      self._compile_thread.join()
      self._compile_thread = None
      self._compile_cancel = None
    while True:
      try:
        self._compiled.get_nowait()
      except queue.Empty:
        break

  def _collect_compiled(self) -> None:
    """Adds freshly compiled slides to the deck and announces them."""
    while True:
      try:
        i, model, path = self._compiled.get_nowait()
      except queue.Empty:
        return
      if i in self._deck:
        continue
      self._deck[i] = (model, path)
      self.handle.send_to_viewer(messages.PreloadModelEvent(model=model))

  def _load_slide(self, index: int) -> None:
    """Loads ``slides[index]``, initializes its state, and notifies viewer."""
    if not self._slides:
      return
    self._index = index % len(self._slides)
    slide = self._slides[self._index]
    if self._index in self._deck:
      model, path = self._deck[self._index]
    else:
      model, path = self.compile_slide(self._index)
      if self._prefetch:
        self._deck[self._index] = (model, path)
    data = mujoco.MjData(model)
    if slide.init_fn is not None:
      slide.init_fn(model, data)
    mujoco.mj_forward(model, data)

    self._is_custom_model = False
    self._slide_loaded_wall_time = time.monotonic()
    self.handle.send_to_viewer(
        SlideChangedEvent(
            index=self._index,
            total=len(self._slides),
            slug=slide.slug,
            category=slide.category,
            title=slide.title,
            caption=slide.caption,
            camera=slide.camera,
            vis_island=slide.vis_island,
            is_custom=False,
        )
    )
    self._loading_slide = True
    try:
      self.handle.set_model(model, path=path, data=data)
    finally:
      self._loading_slide = False

  # Run after StepControl._on_sim_init (Priority.INTERNAL = 1) so initial model
  # pacing is synchronized cleanly.
  @messages.handler(priority=0)
  def _on_sim_init(self, _: plugin.SimInitEvent) -> None:
    if self.handle.model is None and self._slides:
      self._load_slide(self._index)
      self._start_deck_compile()

  @messages.handler
  def _on_select_slide(self, event: SelectSlideEvent) -> bool:
    if not self._slides:
      return True
    if event.index is not None:
      target = event.index % len(self._slides)
    elif self._is_custom_model and event.delta == 0:
      target = self._index
    elif self._is_custom_model:
      # Returning from a dropped custom model resumes at the current slide when
      # pressing Left, or advances when pressing Right.
      target = (
          self._index
          if event.delta < 0
          else (self._index + 1) % len(self._slides)
      )
    else:
      target = (self._index + event.delta) % len(self._slides)
    self._load_slide(target)
    return True

  @messages.handler(priority=0)
  def _on_post_model(self, event: messages.PostModelEvent) -> None:
    same_scene = bool(event.path and event.path == self._loaded_model_path)
    self._loaded_model_path = event.path
    if not same_scene:
      self.handle.dispatch(
          messages.StepControlSnapshot(speed=_scene_default_speed(event.model))
      )
    if self._loading_slide:
      return
    # A model was loaded externally (e.g. drag-and-drop or --model flag).
    self._is_custom_model = True
    self._slide_loaded_wall_time = time.monotonic()
    label = os.path.basename(event.path) if event.path else 'Custom Model'
    self.handle.send_to_viewer(
        SlideChangedEvent(
            index=self._index,
            total=len(self._slides),
            slug='custom',
            category='Custom Scene',
            title=label,
            caption=(
                'Custom model loaded. Drag to orbit or pull bodies, or press'
                ' Left or Right to return to the slideshow.'
            ),
            camera=None,
            vis_island=False,
            is_custom=True,
        )
    )

  # Run after ViewerHandle._on_reset (Priority.INTERNAL = 1) so slide-specific
  # initial joint states (e.g. the swinging fridge door) reapply.
  @messages.handler(priority=0)
  def _on_reset(self, _: messages.ResetEvent) -> None:
    self._slide_loaded_wall_time = time.monotonic()
    if self._is_custom_model or not self._slides:
      return
    slide = self._slides[self._index]
    if (
        slide.init_fn is not None
        and self.handle.model is not None
        and self.handle.data is not None
    ):
      slide.init_fn(self.handle.model, self.handle.data)
      mujoco.mj_forward(self.handle.model, self.handle.data)

  @messages.handler
  def _on_pre_step(self, event: messages.PreStepEvent) -> None:
    self._collect_compiled()
    if not self._is_custom_model and self._slides:
      slide = self._slides[self._index]
      if slide.pre_step_fn is not None:
        slide.pre_step_fn(event.model, event.data)
      if self._autoplay_seconds > 0.0:
        now = time.monotonic()
        if now - self._slide_loaded_wall_time >= self._autoplay_seconds:
          self._load_slide(self._index + 1)

  @messages.handler
  def _on_exit(self, _: messages.ExitEvent) -> None:
    self._stop_deck_compile()

  @messages.handler
  def _on_hot_reload(self, event: messages.HotReloadEvent) -> None:
    if self in event.plugins and not self._is_custom_model and self._slides:
      self._slides = SLIDES
      self._index %= len(self._slides)
      # The edited code may define different scenes: stop the compile thread
      # after the slide it is on, drop the compiled deck and anything it
      # delivered, and compile it again.
      self._stop_deck_compile()
      self._deck.clear()
      self._load_slide(self._index)
      self._start_deck_compile()


_SHORTCUT_ROWS: tuple[tuple[str, str], ...] = (
    ('Left / Right', 'Previous / next slide'),
    ('C / F / I', 'Show contacts / forces / islands'),
    ('F1 / H', 'Show or hide help'),
    ('Space', 'Play / pause'),
    ('Backspace', 'Reset scene'),
    ('Drag body', 'Pull body (Shift: horizontal)'),
)

_RELEASE_NOTES_TITLE = 'MuJoCo 4.0 Interactive Tour'
_RELEASE_NOTES_INTRO = (
    'Each slide runs a live MuJoCo 4.0 scene. Orbit the camera, drag bodies'
    ' with the mouse, scrub the timeline, or switch slides with the dots or'
    ' arrow keys.'
)

# The simulation speed selector: percent of real time, in 5% steps.
_SPEED_STEP_PERCENT = 5.0
_SPEED_STEPS = 20  # 5% .. 100%
_SPEED_SELECTOR_WIDTH = 96.0
_SELECTOR_LABEL_PX = 12.0

# What the card's size was measured for, and (content width, caption height).
_CardMetricsCache = tuple[tuple[object, ...], tuple[float, float]]


def _row_frame_pad_y(shape: OverlayShape) -> float:
  """Vertical frame padding that makes a small text line button-height."""
  return max(0.0, (shape.btn_size - _SELECTOR_LABEL_PX) * 0.5)


def _format_sim_time(seconds: float) -> str:
  """Formats a simulation time compactly, e.g. ``'1.23 s'`` or ``'12.3 s'``."""
  seconds = max(0.0, float(seconds))
  if seconds < 10.0:
    return f'{seconds:.2f} s'
  if seconds < 100.0:
    return f'{seconds:.1f} s'
  return f'{seconds:.0f} s'


def _get_font(io: imgui.IO, index: int = 0) -> imgui.Font | None:
  """Returns ``io.Fonts[index]`` safely, or ``None`` if no fonts are loaded."""
  count = len(io.Fonts)
  if count == 0:
    return None
  return io.Fonts[min(index, count - 1)]


# Draw-list primitives for the card's icons and sliders. Everything is placed
# from its exact center, which ImGui's text-in-button layout cannot promise.


def _u32(color: imgui.Vec4) -> int:
  """Packs ``color`` for the draw list, with the current style alpha applied."""
  return int(imgui.GetColorU32(color))


def _stroke(
    dl: imgui.DrawList,
    points: Sequence[tuple[float, float]],
    col: int,
    thickness: float,
) -> None:
  """Draws a polyline with round caps and joins."""
  vec = [imgui.Vec2(x, y) for x, y in points]
  dl.AddPolyline(vec, col, thickness)
  for p in vec:
    dl.AddCircleFilled(p, thickness * 0.5, col)


def _fill_triangle(
    dl: imgui.DrawList,
    a: tuple[float, float],
    b: tuple[float, float],
    c: tuple[float, float],
    col: int,
) -> None:
  """Fills a triangle; ImGui anti-aliases filled shapes wound clockwise."""
  signed_area = (b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0])
  if signed_area < 0.0:
    b, c = c, b
  dl.AddTriangleFilled(imgui.Vec2(*a), imgui.Vec2(*b), imgui.Vec2(*c), col)


def _draw_chevron(
    dl: imgui.DrawList,
    cx: float,
    cy: float,
    px: float,
    col: int,
    direction: tuple[float, float],
) -> None:
  """Draws a chevron centered on (cx, cy) whose tip points along direction."""
  dx, dy = direction
  along, across = px * 0.25, px * 0.36  # Half extents of the chevron's box.
  tip = (cx + dx * along, cy + dy * along)
  base_x, base_y = cx - dx * along, cy - dy * along
  # The arms end on the far side of the box, offset perpendicular to the tip.
  arm_a = (base_x - dy * across, base_y + dx * across)
  arm_b = (base_x + dy * across, base_y - dx * across)
  _stroke(dl, [arm_a, tip, arm_b], col, px * 0.135)


def _draw_icon(
    dl: imgui.DrawList,
    icon: str,
    cx: float,
    cy: float,
    px: float,
    col: int,
    cutout: int,
) -> None:
  """Draws ``icon`` centered on (``cx``, ``cy``), sized for a ``px`` box.

  Args:
    dl: The draw list to draw on.
    icon: One of the ``ICON_*`` names.
    cx: Center x, in screen pixels.
    cy: Center y, in screen pixels.
    px: Size of the icon's box.
    col: Packed icon color.
    cutout: Packed color of the surface behind the icon, used for parts cut out
      of a solid shape (the help icon's question mark).
  """
  if icon == ICON_PREV:
    _draw_chevron(dl, cx, cy, px, col, (-1.0, 0.0))
  elif icon == ICON_NEXT:
    _draw_chevron(dl, cx, cy, px, col, (1.0, 0.0))
  elif icon == ICON_EXPAND:
    _draw_chevron(dl, cx, cy, px, col, (0.0, -1.0))
  elif icon == ICON_COLLAPSE:
    _draw_chevron(dl, cx, cy, px, col, (0.0, 1.0))
  elif icon == ICON_PLAY:
    w, h = px * 0.6, px * 0.68
    # A triangle's visual center is right of its box center; nudge it over.
    x0 = cx - w * 0.5 + px * 0.05
    _fill_triangle(
        dl, (x0, cy - h * 0.5), (x0 + w, cy), (x0, cy + h * 0.5), col
    )
  elif icon == ICON_PAUSE:
    h, bar, gap = px * 0.66, px * 0.26, px * 0.14
    for x0 in (cx - gap * 0.5 - bar, cx + gap * 0.5):
      dl.AddRectFilled(
          imgui.Vec2(x0, cy - h * 0.5),
          imgui.Vec2(x0 + bar, cy + h * 0.5),
          col,
          px * 0.07,
      )
  elif icon == ICON_RESET:
    # A counter-clockwise circular arrow, open at the upper left where the
    # arrowhead sits. Angles are clockwise from +x (y points down).
    radius, thickness = px * 0.3, px * 0.14
    a_start, a_end = math.radians(190.0), math.radians(-100.0)
    center = imgui.Vec2(cx, cy)
    dl.PathArcTo(center, radius, a_start, a_end)
    dl.PathStroke(col, thickness)
    dl.AddCircleFilled(
        imgui.Vec2(
            cx + radius * math.cos(a_start), cy + radius * math.sin(a_start)
        ),
        thickness * 0.5,
        col,
    )
    end = (cx + radius * math.cos(a_end), cy + radius * math.sin(a_end))
    # Travel direction at the end of a decreasing-angle sweep, and its normal.
    dir_x, dir_y = math.sin(a_end), -math.cos(a_end)
    nx, ny = -dir_y, dir_x
    head_len, head_w = px * 0.32, px * 0.34
    tip = (end[0] + dir_x * head_len * 0.55, end[1] + dir_y * head_len * 0.55)
    bx = end[0] - dir_x * head_len * 0.45
    by = end[1] - dir_y * head_len * 0.45
    _fill_triangle(
        dl,
        tip,
        (bx + nx * head_w * 0.5, by + ny * head_w * 0.5),
        (bx - nx * head_w * 0.5, by - ny * head_w * 0.5),
        col,
    )
  elif icon == ICON_HELP:
    # A solid disk with the question mark cut out of it: a hook from 9
    # o'clock over the top down to a short stem, then a dot, centered.
    dl.AddCircleFilled(imgui.Vec2(cx, cy), px * 0.47, col)
    hook_r, stem, gap, dot_r = px * 0.13, px * 0.09, px * 0.08, px * 0.07
    thickness = px * 0.12
    total = 2.0 * hook_r + stem + gap + 2.0 * dot_r
    top = cy - total * 0.5
    hook_center = imgui.Vec2(cx, top + hook_r)
    dl.PathArcTo(hook_center, hook_r, math.pi, 2.5 * math.pi)
    dl.PathLineTo(imgui.Vec2(cx, top + 2.0 * hook_r + stem))
    dl.PathStroke(cutout, thickness)
    for cap in ((cx - hook_r, top + hook_r), (cx, top + 2.0 * hook_r + stem)):
      dl.AddCircleFilled(imgui.Vec2(*cap), thickness * 0.5, cutout)
    dl.AddCircleFilled(imgui.Vec2(cx, top + total - dot_r), dot_r, cutout)


def _draw_dot(
    dl: imgui.DrawList,
    cx: float,
    cy: float,
    px: float,
    col: int,
    *,
    hollow: bool,
) -> None:
  """Draws a slide dot of diameter ~``px`` centered on (``cx``, ``cy``)."""
  radius = px * 0.44
  if hollow:
    dl.AddCircle(imgui.Vec2(cx, cy), radius - 0.75, col, 0, 1.5)
  else:
    dl.AddCircleFilled(imgui.Vec2(cx, cy), radius, col)


class SlideshowOverlay(plugin.ViewerPlugin):
  """Floating slide card with icon controls, a timeline, and viewport input.

  The card's colors follow the viewer's GUI theme: the Studio app's theme menu
  or, in the web viewer, the theme picked in the browser's role bar, both of
  which arrive as ``plugin.ThemeEvent``.
  """

  _has_studio_app: bool = False
  _card_state: CardState = CardState.EXPANDED
  _show_help: bool = False
  _theme: ux.GuiTheme = DEFAULT_THEME
  _applied_theme: ux.GuiTheme | None = None
  _orbit_deg_per_sec: float = 0.0
  _history_tail_time: float = 0.0
  _status: SlideChangedEvent | None = None

  def __init__(
      self,
      *,
      hide_card: bool = False,
      orbit_deg_per_sec: float = 0.0,
  ) -> None:
    studio_app.install_console_hooks()
    self._has_studio_app = False
    if hide_card:
      self._card_state = CardState.HIDDEN
    else:
      self._card_state = CardState.EXPANDED
    self._show_help = False
    self._theme = DEFAULT_THEME
    self._applied_theme = None
    self._orbit_deg_per_sec = float(orbit_deg_per_sec)
    self._step_control = messages.StepControlSnapshot()
    self._ux_state = ux.UxState()
    self._history_tail_time: float = 0.0
    self._model_path: str = ''
    self._dragging_body: bool = False
    self._card_metrics_cache: _CardMetricsCache | None = None
    self._last_mouse_pos: tuple[float, float] | None = None
    self._last_mouse_bottom_t: float = -100.0
    self._card_hovered_last_frame: bool = False
    self._status = SlideChangedEvent(
        index=0,
        total=len(SLIDES),
        slug=SLIDES[0].slug,
        category=SLIDES[0].category,
        title=SLIDES[0].title,
        caption=SLIDES[0].caption,
        camera=SLIDES[0].camera,
        vis_island=SLIDES[0].vis_island,
    )

  @property
  def theme(self) -> ux.GuiTheme:
    return self._theme

  @property
  def card_state(self) -> CardState:
    return self._card_state

  @property
  def hide_card(self) -> bool:
    return self._card_state == CardState.HIDDEN

  @property
  def is_dark(self) -> bool:
    return self._theme != ux.GuiTheme.LIGHT

  def _get_theme_spec(self) -> ThemeSpec:
    return THEMES.get(self._theme, THEMES[DEFAULT_THEME])

  def _apply_theme_if_needed(self) -> None:
    """Matches Studio's base style (tooltips, popups) to the card's theme.

    Skipped inside the Studio app, which owns its own theme.
    """
    if self._has_studio_app:
      return
    if self._theme != self._applied_theme and imgui.GetCurrentContext() != 0:
      ux.setup_theme(self._theme)
      self._applied_theme = self._theme

  def set_theme(self, theme: ux.GuiTheme) -> None:
    """Switches the card to the look that goes with a Studio GUI theme."""
    self._theme = theme
    self._apply_theme_if_needed()

  def cycle_card_state(self) -> CardState:
    """Cycles ``EXPANDED -> COMPACT -> HIDDEN -> EXPANDED`` (bound to ``H``)."""
    order = (CardState.EXPANDED, CardState.COMPACT, CardState.HIDDEN)
    idx = order.index(self._card_state)
    self._card_state = order[(idx + 1) % len(order)]
    return self._card_state

  def toggle_compact_card(self) -> CardState:
    """Toggles between ``EXPANDED`` and ``COMPACT`` via the chevron button."""
    if self._card_state == CardState.EXPANDED:
      self._card_state = CardState.COMPACT
    else:
      self._card_state = CardState.EXPANDED
    return self._card_state

  def toggle_help(self) -> bool:
    """Toggles the centered release notes and help overlay."""
    self._show_help = not self._show_help
    return self._show_help

  @messages.handler
  def on_studio_app_init(self, event: studio_app.InitEvent) -> None:
    del event
    self._has_studio_app = True

  @messages.handler
  def on_viewer_init(self, _: plugin.ViewerInitEvent) -> None:
    studio_app.install_console_hooks()
    if not self._has_studio_app:
      self._apply_theme_if_needed()
      preset = self._status.camera if self._status is not None else None
      apply_interactive_camera(
          self.viewer.model, self.viewer.data, self.viewer.camera, preset
      )

  # The web viewer (and the Studio app) apply the theme to ImGui themselves
  # before dispatching; this only switches the card's colors to match.
  @messages.handler
  def on_theme(self, event: plugin.ThemeEvent) -> None:
    self.set_theme(event.theme)

  @messages.handler
  def on_slide_changed(self, event: SlideChangedEvent) -> None:
    self._status = event

  # Run after studio_app.App._on_post_model (Priority.INTERNAL = 1) so the
  # slide's interactive free-camera framing takes precedence in both modes.
  @messages.handler(priority=0)
  def on_post_model(self, event: messages.PostModelEvent) -> None:
    self._ux_state = ux.UxState()
    self._history_tail_time = 0.0
    mujoco.mjv_defaultPerturb(self.viewer.perturb)
    self._dragging_body = False
    self._sync_step_control_for_scene(event.model, event.path)
    status = self._status
    preset = status.camera if status is not None else None
    apply_interactive_camera(
        event.model, event.data, self.viewer.camera, preset
    )
    self.viewer.vis_options.flags[mujoco.mjtVisFlag.mjVIS_ISLAND] = (
        1 if (status is not None and status.vis_island) else 0
    )

  def _sync_step_control_for_scene(
      self, model: mujoco.MjModel, path: str
  ) -> None:
    """Keeps the viewer's step-control mirror in step with the sim's.

    The sim's StepControl plugin resets on every new scene and keeps its
    settings only when the same file is reloaded, without announcing either. A
    stale mirror here would, for example, treat drags as paused-mode pose edits
    while the sim is running. Mirror the sim at ``model``'s authored
    ``<global realtime=.../>`` speed and notify the sim so both stay aligned.

    Args:
      model: The loaded model.
      path: The new model's file path, or empty.
    """
    same_scene = bool(path and path == self._model_path)
    self._model_path = path
    if same_scene:
      return
    self._step_control = messages.StepControlSnapshot(
        speed=_scene_default_speed(model)
    )
    self.viewer.send_to_sim(self._step_control)

  @messages.handler(priority=messages.Priority.INTERNAL)
  def on_step_control(self, event: messages.StepControlSnapshot) -> None:
    self._step_control = event

  @messages.handler(priority=messages.Priority.INTERNAL)
  def on_pause(self, event: messages.RequestPauseEvent) -> None:
    self._step_control = dataclasses.replace(
        self._step_control, pause_state=event.pause_state
    )

  @messages.handler(priority=messages.Priority.INTERNAL)
  def on_history(self, event: sim_history.SimHistorySnapshot) -> None:
    """Mirrors the sim's history buffer (size, index, head and tail times)."""
    self._ux_state.history_size = event.size
    self._ux_state.history_index = event.index
    self._ux_state.sim_head_time = event.head_time
    self._history_tail_time = event.tail_time

  def _scrub_history(self, index: int) -> None:
    """Asks the sim to show history frame ``index`` (0 = newest), pausing."""
    self._ux_state.history_index = index
    self.viewer.send_to_sim(
        sim_history.SimHistorySnapshot(index=self._ux_state.history_index)
    )

  def _request_slide(self, delta: int) -> None:
    self.viewer.send_to_sim(SelectSlideEvent(delta=delta))

  def _toggle_pause(self) -> None:
    if self._step_control.pause_state == sim.PauseState.UNPAUSED:
      next_state = sim.PauseState.NORMAL_PAUSED
    else:
      next_state = sim.PauseState.UNPAUSED
    self._step_control = dataclasses.replace(
        self._step_control, pause_state=next_state
    )
    self.viewer.send_to_sim(messages.RequestPauseEvent(pause_state=next_state))

  def _reset_current_scene(self) -> None:
    mujoco.mj_resetData(self.viewer.model, self.viewer.data)
    mujoco.mj_forward(self.viewer.model, self.viewer.data)
    self.viewer.send_to_sim(messages.ResetEvent())
    self.viewer.get_sim_snapshots()

  def _set_speed(self, speed_percent: float) -> None:
    """Sets the simulation speed as a percentage of real time."""
    self._step_control = dataclasses.replace(
        self._step_control, speed=float(speed_percent)
    )
    self.viewer.send_to_sim(self._step_control)

  def _handle_body_drag(self) -> bool:
    """Pulls a dynamic body after the mouse while the left button is held.

    A left press on a body that can move (see ``is_draggable_body``) selects
    it and, until the button is released, pulls it with the usual perturbation
    spring; Shift pulls in the horizontal plane instead of the vertical one.
    Presses on the floor, on static bodies, or on empty space clear the
    selection and fall through to the camera controls. Ctrl keeps Studio's
    perturbation bindings.

    Returns:
      True while a body is being dragged, so the caller skips camera input.
    """
    io = imgui.GetIO()
    perturb = self.viewer.perturb
    if (
        io.WantCaptureMouse
        or io.KeyCtrl
        or io.DisplaySize.x <= 0
        or io.DisplaySize.y <= 0
    ):
      self._dragging_body = False
      return False

    model, data = self.viewer.model, self.viewer.data
    if imgui.IsMouseClicked(imgui.MouseButton.Left):
      picked = ux.Pick(
          model,
          data,
          self.viewer.camera,
          io.MousePos.x / io.DisplaySize.x,
          io.MousePos.y / io.DisplaySize.y,
          io.DisplaySize.x / io.DisplaySize.y,
          self.viewer.vis_options,
      )
      self._dragging_body = is_draggable_body(model, picked.body)
      if self._dragging_body:
        perturb.select = picked.body
        perturb.flexselect = picked.flex
        perturb.skinselect = picked.skin
        offset = (
            np.asarray(picked.point, dtype=np.float64) - data.xpos[picked.body]
        )
        xmat = np.asarray(data.xmat[picked.body], dtype=np.float64).reshape(
            3, 3
        )
        perturb.localpos = xmat.T @ offset
        ux.InitPerturb(
            model,
            data,
            self.viewer.camera,
            perturb,
            int(mujoco.mjtPertBit.mjPERT_TRANSLATE),
        )
      else:
        perturb.select = 0
        perturb.flexselect = -1
        perturb.skinselect = -1

    if not self._dragging_body:
      return False
    if not imgui.IsMouseDown(imgui.MouseButton.Left):
      self._dragging_body = False
      perturb.active = 0
      return False

    dx = io.MouseDelta.x / io.DisplaySize.x
    dy = io.MouseDelta.y / io.DisplaySize.y
    if dx != 0.0 or dy != 0.0:
      action = (
          mujoco.mjtMouse.mjMOUSE_MOVE_H
          if io.KeyShift
          else mujoco.mjtMouse.mjMOUSE_MOVE_V
      )
      ux.MovePerturb(
          model, data, self.viewer.camera, perturb, int(action), dx, dy
      )
    return True

  def _handle_minimal_inputs(self) -> None:
    """Handles mouse and minimal slide/sim keys when running without App."""
    drop_file = self.viewer.get_drop_file()
    if drop_file:
      self.viewer.send_to_sim(messages.LoadModelEvent(path=drop_file))

    # Dragging a dynamic body pulls it; any other drag drives the camera.
    if not self._handle_body_drag():
      inputs.handle_mouse_events(
          self.viewer.model,
          self.viewer.data,
          self.viewer.camera,
          self.viewer.vis_options,
          self.viewer.perturb,
          self._ux_state,
      )

    is_paused = self._step_control.pause_state == sim.PauseState.NORMAL_PAUSED
    viewer_utils.apply_perturb(
        self.viewer, self.viewer.model, self.viewer.data, is_paused
    )

    io = imgui.GetIO()
    if io.WantCaptureKeyboard:
      return

    pressed = imgui.IsKeyChordPressed
    if (
        pressed(imgui.Key.H)
        or pressed(imgui.Key.F1)
        or pressed(imgui.Key.Escape)
    ):
      self.toggle_help()
    elif pressed(imgui.Key.LeftArrow):
      self._request_slide(-1)
    elif pressed(imgui.Key.RightArrow):
      self._request_slide(1)
    elif pressed(imgui.Key.Space):
      self._toggle_pause()
    elif pressed(imgui.Key.Backspace):
      self._reset_current_scene()
    elif pressed(imgui.Key.I):
      self.viewer.vis_options.flags[mujoco.mjtVisFlag.mjVIS_ISLAND] ^= 1
    elif pressed(imgui.Key.C):
      self.viewer.vis_options.flags[mujoco.mjtVisFlag.mjVIS_CONTACTPOINT] ^= 1
    elif pressed(imgui.Key.F):
      self.viewer.vis_options.flags[mujoco.mjtVisFlag.mjVIS_CONTACTFORCE] ^= 1

  def _handle_studio_app_shortcuts(self) -> None:
    """Handles slide/help shortcuts when embedded inside Studio App."""
    if imgui.GetIO().WantCaptureKeyboard:
      return
    pressed = imgui.IsKeyChordPressed
    if (
        pressed(imgui.Key.H)
        or pressed(imgui.Key.F1)
        or pressed(imgui.Key.Escape)
    ):
      self.toggle_help()
      return
    is_paused = self._step_control.pause_state == sim.PauseState.NORMAL_PAUSED
    if not is_paused:
      if pressed(imgui.Key.LeftArrow):
        self._request_slide(-1)
      elif pressed(imgui.Key.RightArrow):
        self._request_slide(1)

  def _apply_auto_orbit(self) -> None:
    if self._orbit_deg_per_sec == 0.0:
      return
    if self.viewer.camera.type != int(mujoco.mjtCamera.mjCAMERA_FREE):
      return
    any_mouse_down = (
        imgui.IsMouseDown(imgui.MouseButton.Left)
        or imgui.IsMouseDown(imgui.MouseButton.Right)
        or imgui.IsMouseDown(imgui.MouseButton.Middle)
    )
    if any_mouse_down:
      return
    dt = float(imgui.GetIO().DeltaTime)
    if 0.0 < dt < 0.25:
      self.viewer.camera.azimuth = (
          float(self.viewer.camera.azimuth) + self._orbit_deg_per_sec * dt
      ) % 360.0

  @messages.handler
  def on_update(self, _: messages.UpdateEvent) -> None:
    if self._has_studio_app:
      self._handle_studio_app_shortcuts()
    else:
      self._handle_minimal_inputs()
    self._apply_auto_orbit()

  def _resolve_palette_and_shape(
      self,
  ) -> tuple[OverlayPalette, OverlayShape]:
    spec = self._get_theme_spec()
    return spec.palette, spec.shape

  def _dot_color(
      self, index: int, total: int, pal: OverlayPalette
  ) -> imgui.Vec4:
    ramp = pal.dot_ramp
    if not ramp:
      return pal.accent
    if len(ramp) == 1 or total <= 1:
      return ramp[0]
    pos = (index / float(total - 1)) * (len(ramp) - 1)
    seg = min(int(pos), len(ramp) - 2)
    return _mix(ramp[seg], ramp[seg + 1], pos - seg)

  def _tooltip(
      self,
      pal: OverlayPalette,
      *,
      text: str = '',
      keys: Sequence[tuple[str, str]] = (),
      instant: bool = False,
  ) -> None:
    """Shows a tooltip for the last item, after the hover delay or at once.

    Args:
      pal: The theme's colors.
      text: One-line label (for items without shortcuts, such as slide dots).
      keys: ``(key, action)`` rows for the borderless shortcut table.
      instant: Show at once rather than after ImGui's hover delay.
    """
    if not text and not keys:
      return
    if not imgui.IsItemHovered():
      return
    if not instant and not imgui.IsItemHovered(
        int(imgui.HoveredFlags.ForTooltip)
    ):
      return
    io = imgui.GetIO()
    font0 = _get_font(io, 0)
    imgui.PushStyleColor(imgui.Col.PopupBg, pal.tooltip_bg)
    imgui.PushStyleColor(imgui.Col.Text, pal.tooltip_text)
    imgui.PushStyleColor(imgui.Col.Border, pal.outline)
    try:
      if imgui.BeginTooltip():
        try:
          if font0 is not None:
            imgui.PushFont(font0, 13.0)
          try:
            if text:
              imgui.TextUnformatted(text)
            if keys and imgui.BeginTable('##SlideshowTooltipKeys', 2):
              for key_label, action in keys:
                imgui.TableNextRow()
                imgui.TableNextColumn()
                imgui.TextColored(pal.accent_text, key_label)
                imgui.TableNextColumn()
                imgui.TextUnformatted(f'  {action}')
              imgui.EndTable()
          finally:
            if font0 is not None:
              imgui.PopFont()
        finally:
          imgui.EndTooltip()
    finally:
      imgui.PopStyleColor(3)

  def _icon_button(
      self,
      icon: str,
      sid: str,
      pal: OverlayPalette,
      shape: OverlayShape,
      *,
      text: str = '',
      keys: Sequence[tuple[str, str]] = (),
      primary: bool = False,
  ) -> bool:
    """Draws a round icon button.

    An invisible button takes the input; the circle and the icon are drawn on
    the draw list from the button's exact center.

    Args:
      icon: One of the ``ICON_*`` names.
      sid: Unique id of the button within the card.
      pal: The theme's colors.
      shape: The theme's geometry.
      text: One-line tooltip label; see ``_tooltip``.
      keys: Shortcut rows for the tooltip; see ``_tooltip``.
      primary: Whether to draw the filled, emphasized variant.

    Returns:
      True when the button was clicked.
    """
    size = imgui.Vec2(shape.btn_size, shape.btn_size)
    clicked = bool(imgui.InvisibleButton(f'##{sid}', size))
    hovered = bool(imgui.IsItemHovered())
    active = bool(imgui.IsItemActive())
    p_min = imgui.GetItemRectMin()
    radius = shape.btn_size * 0.5
    center = imgui.Vec2(p_min.x + radius, p_min.y + radius)

    if primary:
      fg = pal.on_container
      if active:
        bg = _mix(pal.container, pal.on_container, 0.14)
      elif hovered:
        bg = _mix(pal.container, pal.on_container, 0.08)
      else:
        bg = pal.container
    else:
      fg = pal.accent if hovered else pal.on_surface
      if active:
        bg = pal.press
      elif hovered:
        bg = pal.hover
      else:
        bg = None

    dl = imgui.GetWindowDrawList()
    if bg is not None:
      dl.AddCircleFilled(center, radius, _u32(bg))
    # What shows through a cut-out: the card, or the button's circle on it.
    behind = pal.surface if bg is None else _over(pal.surface, bg)
    _draw_icon(
        dl, icon, center.x, center.y, shape.icon_px, _u32(fg), _u32(behind)
    )

    if hovered:
      imgui.SetMouseCursor(imgui.MouseCursor.Hand)
    if text or keys:
      self._tooltip(pal, text=text, keys=keys)
    return clicked

  def _draw_progress_dots(
      self,
      io: imgui.IO,
      status: SlideChangedEvent,
      pal: OverlayPalette,
      shape: OverlayShape,
  ) -> None:
    if io.DisplaySize.x < 600.0:
      mono = _get_font(io, 1)
      if mono is not None:
        imgui.PushFont(mono, 12.0)
      try:
        imgui.AlignTextToFramePadding()
        if status.is_custom:
          imgui.TextColored(pal.muted, 'CUSTOM')
        else:
          imgui.TextColored(
              pal.muted, f'{status.index + 1:02d} / {status.total:02d}'
          )
      finally:
        if mono is not None:
          imgui.PopFont()
      return

    total = status.total
    dl = imgui.GetWindowDrawList()
    for i in range(total):
      clicked = bool(
          imgui.InvisibleButton(
              f'##dot{i}', imgui.Vec2(shape.dot_w, shape.btn_size)
          )
      )
      hovered = bool(imgui.IsItemHovered())
      if (i == status.index) and not status.is_custom:
        col = self._dot_color(i, total, pal)
        px = shape.dot_px_active
      elif hovered:
        col = pal.on_surface
        px = shape.dot_px
      else:
        col = pal.dot_idle
        px = shape.dot_px
      p_min = imgui.GetItemRectMin()
      _draw_dot(
          dl,
          p_min.x + shape.dot_w * 0.5,
          p_min.y + shape.btn_size * 0.5,
          px,
          _u32(col),
          hollow=status.is_custom,
      )

      if clicked:
        self.viewer.send_to_sim(SelectSlideEvent(index=i))
      if hovered:
        imgui.SetMouseCursor(imgui.MouseCursor.Hand)
      if i < len(SLIDES):
        tip = f'{i + 1}. {SLIDES[i].title}'
      else:
        tip = f'Slide {i + 1}'
      self._tooltip(pal, text=tip, instant=True)
      if i < total - 1:
        imgui.SameLine(0.0, 0.0)

  # Card layout. Rows size themselves from these widths rather than from the
  # available region: the window auto-sizes to its content, so using the
  # available width would make the card grow every frame.

  def _nav_width(
      self, io: imgui.IO, status: SlideChangedEvent, shape: OverlayShape
  ) -> float:
    """Width of the slide navigation: prev, the dots, next."""
    if io.DisplaySize.x >= 600.0:
      mid_w = float(status.total) * shape.dot_w
    else:
      mid_w = 56.0
    return 2.0 * shape.btn_size + 2.0 * 4.0 + mid_w

  def _controls_width(self, shape: OverlayShape) -> float:
    """Width of play/pause, the speed selector, and reset."""
    return 2.0 * shape.btn_size + 2.0 * 8.0 + _SPEED_SELECTOR_WIDTH

  def _chrome_width(self, shape: OverlayShape) -> float:
    """Width of the help and collapse buttons."""
    return 2.0 * shape.btn_size + 4.0

  def _time_label_width(self, io: imgui.IO, shape: OverlayShape) -> float:
    mono = _get_font(io, 1)
    if mono is not None:
      imgui.PushFont(mono, shape.eyebrow_px)
    try:
      return float(imgui.CalcTextSize('000.00 s').x)
    finally:
      if mono is not None:
        imgui.PopFont()

  def _card_metrics(
      self, io: imgui.IO, status: SlideChangedEvent, shape: OverlayShape
  ) -> tuple[float, float]:
    """Returns the card's content width and caption height.

    Both fit every slide of the deck (and the current custom model, if any),
    so the card keeps one size as the slides change: wide enough for the
    widest title next to the slide navigation and for the control row with a
    usable timeline, and tall enough for the longest caption at that width.
    The measurements are cached until something they depend on changes.

    Args:
      io: The ImGui IO, for fonts.
      status: The current slide.
      shape: The theme's geometry.
    """
    wide = io.DisplaySize.x >= 600.0
    custom = (status.title, status.caption) if status.is_custom else ('', '')
    key = (self._theme, wide, status.total, float(io.FontGlobalScale))
    key += (len(io.Fonts),) + custom
    if self._card_metrics_cache is not None:
      cached_key, metrics = self._card_metrics_cache
      if cached_key == key:
        return metrics

    titles = [slide.title for slide in SLIDES]
    captions = [slide.caption for slide in SLIDES]
    if status.is_custom:
      titles.append(status.title)
      captions.append(status.caption)

    font0 = _get_font(io, 0)
    if font0 is not None:
      imgui.PushFont(font0, shape.title_px)
    try:
      title_w = max(float(imgui.CalcTextSize(t).x) for t in titles)
    finally:
      if font0 is not None:
        imgui.PopFont()
    header_w = title_w + 24.0 + self._nav_width(io, status, shape)
    timeline_min_w = (self._time_label_width(io, shape) + 8.0) + 200.0
    widgets_w = (
        self._controls_width(shape)
        + shape.row_gap
        + timeline_min_w
        + shape.row_gap
        + self._chrome_width(shape)
    )
    content_w = max(shape.wrap_width, header_w, widgets_w)

    if font0 is not None:
      imgui.PushFont(font0, shape.caption_px)
    try:
      caption_h = max(
          float(imgui.CalcTextSize(c, None, False, content_w).y)
          for c in captions
      )
    finally:
      if font0 is not None:
        imgui.PopFont()

    self._card_metrics_cache = (key, (content_w, caption_h))
    return content_w, caption_h

  def _draw_header(
      self,
      io: imgui.IO,
      status: SlideChangedEvent,
      pal: OverlayPalette,
      shape: OverlayShape,
      content_w: float,
  ) -> None:
    """Draws the title on the left and the slide navigation on the right."""
    font0 = _get_font(io, 0)
    row_x = float(imgui.GetCursorPosX())
    row_y = float(imgui.GetCursorPosY())
    # An invisible full-height item first, so the row is as tall as the
    # buttons wherever the title is placed within it.
    imgui.Dummy(imgui.Vec2(0.0, shape.btn_size))
    imgui.SameLine(0.0, 0.0)
    imgui.SetCursorPosY(row_y + (shape.btn_size - shape.title_px) * 0.5)
    imgui.PushStyleColor(imgui.Col.Text, pal.on_surface)
    if font0 is not None:
      imgui.PushFont(font0, shape.title_px)
    try:
      imgui.TextUnformatted(status.title)
    finally:
      if font0 is not None:
        imgui.PopFont()
      imgui.PopStyleColor()
    imgui.SameLine(0.0, 0.0)
    imgui.SetCursorPos(
        imgui.Vec2(
            row_x + content_w - self._nav_width(io, status, shape), row_y
        )
    )
    self._draw_nav(io, status, pal, shape)

  def _draw_nav(
      self,
      io: imgui.IO,
      status: SlideChangedEvent,
      pal: OverlayPalette,
      shape: OverlayShape,
  ) -> None:
    """Draws prev, the clickable slide dots, and next."""
    if status.is_custom:
      prev_tip = next_tip = 'Back to the slideshow'
    else:
      prev_tip, next_tip = 'Previous slide', 'Next slide'
    if self._icon_button(
        ICON_PREV, 'prev', pal, shape, keys=(('Left', prev_tip),)
    ):
      self._request_slide(-1)
    imgui.SameLine(0.0, 4.0)
    self._draw_progress_dots(io, status, pal, shape)
    imgui.SameLine(0.0, 4.0)
    if self._icon_button(
        ICON_NEXT, 'next', pal, shape, keys=(('Right', next_tip),)
    ):
      self._request_slide(1)

  def _draw_caption(
      self,
      io: imgui.IO,
      status: SlideChangedEvent,
      pal: OverlayPalette,
      shape: OverlayShape,
      content_w: float,
      caption_h: float,
  ) -> None:
    """Draws the feature description, wrapped to the card's width.

    The caption area is ``caption_h`` tall whatever the text's length, so the
    card keeps one height across slides.

    Args:
      io: The ImGui IO, for fonts.
      status: The current slide.
      pal: The theme's colors.
      shape: The theme's geometry.
      content_w: The card's content width, which the text wraps to.
      caption_h: Height reserved for the caption; see ``_card_metrics``.
    """
    font0 = _get_font(io, 0)
    top_y = float(imgui.GetCursorPosY())
    imgui.PushStyleColor(imgui.Col.Text, pal.on_surface_var)
    if font0 is not None:
      imgui.PushFont(font0, shape.caption_px)
    imgui.PushTextWrapPos(imgui.GetCursorPosX() + content_w)
    try:
      imgui.TextUnformatted(status.caption)
    finally:
      imgui.PopTextWrapPos()
      if font0 is not None:
        imgui.PopFont()
      imgui.PopStyleColor()
    spacing_y = float(imgui.GetStyle().ItemSpacing.y)
    imgui.SetCursorPosY(
        max(float(imgui.GetCursorPosY()), top_y + caption_h + spacing_y)
    )

  def _draw_history_scrubber(
      self,
      io: imgui.IO,
      pal: OverlayPalette,
      shape: OverlayShape,
      track_w: float,
      row_y: float,
  ) -> None:
    """Draws the simulation history timeline slider and timestamps.

    Dragging the slider asks the sim to show that recorded frame and pauses;
    playing resumes from there. Index 0 is the newest frame. The slider is a
    float slider so the grab keeps one size however many frames there are.

    Args:
      io: The ImGui IO, for fonts.
      pal: The theme's colors.
      shape: The theme's geometry.
      track_w: Width of the slider track.
      row_y: Top of the button-height row the timeline is centered in.
    """
    mono = _get_font(io, 1)
    size = int(self._ux_state.history_size)
    index = int(self._ux_state.history_index) if size > 1 else 0
    head_time = float(self._ux_state.sim_head_time)
    tail_time = float(self._history_tail_time)
    now = (
        float(self.viewer.data.time)
        if self.viewer.data is not None
        else head_time
    )
    oldest_text = _format_sim_time(tail_time)
    current_text = _format_sim_time(now)
    slider_h = shape.eyebrow_px + 2.0 * 2.0
    slider_y = row_y + (shape.btn_size - slider_h) * 0.5
    label_w = self._time_label_width(io, shape)
    dl = imgui.GetWindowDrawList()

    def time_label(text: str, *, align_right: bool) -> None:
      # A row-height placeholder holds the space; the text is drawn centered
      # on it, outside ImGui's text baseline alignment.
      imgui.Dummy(imgui.Vec2(label_w, shape.btn_size))
      p_min = imgui.GetItemRectMin()
      text_size = imgui.CalcTextSize(text)
      x = p_min.x + (label_w - text_size.x if align_right else 0.0)
      y = p_min.y + (shape.btn_size - text_size.y) * 0.5
      dl.AddText(imgui.Vec2(x, y), _u32(pal.muted), text)

    if mono is not None:
      imgui.PushFont(mono, shape.eyebrow_px)
    imgui.PushStyleColor(imgui.Col.FrameBg, pal.hover)
    imgui.PushStyleColor(imgui.Col.FrameBgHovered, pal.press)
    imgui.PushStyleColor(imgui.Col.FrameBgActive, pal.press)
    imgui.PushStyleColor(imgui.Col.SliderGrab, pal.accent)
    imgui.PushStyleColor(imgui.Col.SliderGrabActive, pal.accent)
    # A capsule track with a round grab inset by the frame padding.
    imgui.PushStyleVar(imgui.StyleVar.FrameRounding, slider_h * 0.5)
    imgui.PushStyleVar(imgui.StyleVar.GrabRounding, slider_h * 0.5 - 2.0)
    imgui.PushStyleVar(imgui.StyleVar.GrabMinSize, slider_h - 4.0)
    imgui.PushStyleVar(imgui.StyleVar.FramePadding, imgui.Vec2(4.0, 2.0))
    try:
      imgui.SetCursorPosY(slider_y)
      imgui.SetNextItemWidth(track_w)
      min_idx = float(-max(size - 1, 1))
      changed, value = imgui.SliderFloat(
          '##SlideshowHistory',
          float(index),
          min_idx,
          0.0,
          '',
          int(imgui.SliderFlags.AlwaysClamp),
      )
      p_min = imgui.GetItemRectMin()
      p_max = imgui.GetItemRectMax()
      dl.AddRect(p_min, p_max, _u32(pal.outline), slider_h * 0.5, 1.0)
      if mono is not None:
        imgui.PushFont(mono, _SELECTOR_LABEL_PX)
      try:
        text_size = imgui.CalcTextSize(oldest_text)
        text_pos = imgui.Vec2(
            p_min.x + 8.0,
            p_min.y + (slider_h - text_size.y) * 0.5,
        )
        grab_r = slider_h * 0.5 - 2.0
        t_grab = (
            (float(index) - min_idx) / max(1.0, -min_idx) if size > 1 else 1.0
        )
        x_first = p_min.x + slider_h * 0.5
        x_last = p_max.x - slider_h * 0.5
        grab_x = x_first + t_grab * (x_last - x_first)
        grab_min_x, grab_max_x = grab_x - grab_r, grab_x + grab_r
        on_accent = imgui.Vec4(pal.surface.x, pal.surface.y, pal.surface.z, 1.0)
        for clip_x0, clip_x1, color in (
            (p_min.x, grab_min_x, pal.muted),
            (grab_min_x, grab_max_x, on_accent),
            (grab_max_x, p_max.x, pal.muted),
        ):
          if clip_x1 <= clip_x0:
            continue
          imgui.PushClipRect(
              imgui.Vec2(clip_x0, p_min.y), imgui.Vec2(clip_x1, p_max.y), True
          )
          dl.AddText(text_pos, _u32(color), oldest_text)
          imgui.PopClipRect()
      finally:
        if mono is not None:
          imgui.PopFont()
      if size > 1:
        self._tooltip(
            pal,
            text=(
                f'History [{oldest_text}, {current_text}]\n'
                'Drag to rewind; press play to resume'
            ),
        )
        if imgui.IsItemHovered():
          imgui.SetMouseCursor(imgui.MouseCursor.Hand)
        if changed and int(round(value)) != int(self._ux_state.history_index):
          self._scrub_history(int(round(value)))
      imgui.SameLine(0.0, 8.0)
      imgui.SetCursorPosY(row_y)
      time_label(current_text, align_right=False)
    finally:
      imgui.PopStyleVar(4)
      imgui.PopStyleColor(5)
      if mono is not None:
        imgui.PopFont()

  def _draw_speed_selector(
      self, io: imgui.IO, pal: OverlayPalette, shape: OverlayShape
  ) -> None:
    """Draws the speed selector: 5% to 100% of real time in 5% steps.

    Styled like the timeline: a thin outlined capsule with a round accent
    grab. The number is the requested speed; a grey bar from the left shows
    the speed the sim achieves, so it reaches the grab when the sim keeps up
    and stops short when it cannot. Clicking or dragging anywhere on the
    track picks the nearest step.

    Args:
      io: The ImGui IO, for fonts.
      pal: The theme's colors.
      shape: The theme's geometry.
    """
    mono = _get_font(io, 1)
    speed = float(self._step_control.speed)
    current = min(_SPEED_STEPS, max(1, int(round(speed / _SPEED_STEP_PERCENT))))
    is_paused = self._step_control.pause_state != sim.PauseState.UNPAUSED
    measured = float(self._step_control.speed_measured)
    width, height = _SPEED_SELECTOR_WIDTH, shape.eyebrow_px + 2.0 * 2.0
    radius = height * 0.5
    grab_r = radius - 2.0  # Inset like the timeline's grab.
    imgui.InvisibleButton('##SlideshowSpeed', imgui.Vec2(width, height))
    hovered = bool(imgui.IsItemHovered())
    active = bool(imgui.IsItemActive())
    p_min = imgui.GetItemRectMin()
    p_max = imgui.GetItemRectMax()
    cy = p_min.y + radius
    x_first, x_last = p_min.x + radius, p_max.x - radius  # Grab centers.

    if active:
      t = (float(imgui.GetMousePos().x) - x_first) / max(1.0, x_last - x_first)
      chosen = 1 + int(round(min(1.0, max(0.0, t)) * (_SPEED_STEPS - 1)))
      if chosen != current:
        self._set_speed(chosen * _SPEED_STEP_PERCENT)
        current = chosen

    dl = imgui.GetWindowDrawList()
    track = pal.press if (hovered or active) else pal.hover
    dl.AddRectFilled(p_min, p_max, _u32(track), radius)
    show_measured = measured >= 0.0 and not is_paused
    if show_measured:
      t_measured = min(1.0, max(0.0, measured / 100.0))
      bar_end = x_first + t_measured * (x_last - x_first)
      bar = imgui.Vec4(pal.muted.x, pal.muted.y, pal.muted.z, 0.45)
      dl.AddRectFilled(
          imgui.Vec2(p_min.x + 2.0, p_min.y + 2.0),
          imgui.Vec2(max(p_min.x + 2.0 + 2.0 * grab_r, bar_end), p_max.y - 2.0),
          _u32(bar),
          grab_r,
      )
    t = (current - 1) / float(_SPEED_STEPS - 1)
    grab_x = x_first + t * (x_last - x_first)
    dl.AddCircleFilled(imgui.Vec2(grab_x, cy), grab_r, _u32(pal.accent))
    dl.AddRect(p_min, p_max, _u32(pal.outline), radius, 1.0)

    # The label is centered on the track; where the grab passes under it the
    # text switches to the card color so it stays readable on the accent.
    label = f'{current * _SPEED_STEP_PERCENT:.0f}%'
    on_accent = imgui.Vec4(pal.surface.x, pal.surface.y, pal.surface.z, 1.0)
    if mono is not None:
      imgui.PushFont(mono, _SELECTOR_LABEL_PX)
    try:
      text_size = imgui.CalcTextSize(label)
      text_pos = imgui.Vec2(
          p_min.x + (width - text_size.x) * 0.5,
          p_min.y + (height - text_size.y) * 0.5,
      )
      grab_min_x, grab_max_x = grab_x - grab_r, grab_x + grab_r
      for clip_x0, clip_x1, color in (
          (p_min.x, grab_min_x, pal.on_surface),
          (grab_min_x, grab_max_x, on_accent),
          (grab_max_x, p_max.x, pal.on_surface),
      ):
        if clip_x1 <= clip_x0:
          continue
        imgui.PushClipRect(
            imgui.Vec2(clip_x0, p_min.y), imgui.Vec2(clip_x1, p_max.y), True
        )
        dl.AddText(text_pos, _u32(color), label)
        imgui.PopClipRect()
    finally:
      if mono is not None:
        imgui.PopFont()

    desired_pct = current * _SPEED_STEP_PERCENT
    if is_paused:
      achieved_pct = 0.0
    elif measured >= 0.0:
      achieved_pct = measured
    else:
      achieved_pct = desired_pct
    self._tooltip(
        pal,
        text=(
            f'Target speed: {desired_pct:.0f}%\n'
            f'Actual speed: {achieved_pct:.0f}%'
        ),
    )
    if hovered:
      imgui.SetMouseCursor(imgui.MouseCursor.Hand)

  def _draw_controls(
      self,
      io: imgui.IO,
      pal: OverlayPalette,
      shape: OverlayShape,
      row_y: float,
  ) -> None:
    """Draws play/pause, the speed selector, and reset.

    Args:
      io: The ImGui IO.
      pal: The theme's colors.
      shape: The theme's geometry.
      row_y: Top of the button-height row, to center the thin speed selector.
    """
    is_paused = self._step_control.pause_state != sim.PauseState.UNPAUSED
    play_icon = ICON_PLAY if is_paused else ICON_PAUSE
    if self._icon_button(
        play_icon,
        'play_pause',
        pal,
        shape,
        keys=(('Space', 'Play / pause'),),
        primary=True,
    ):
      self._toggle_pause()
    imgui.SameLine(0.0, 8.0)
    selector_h = shape.eyebrow_px + 2.0 * 2.0
    imgui.SetCursorPosY(row_y + (shape.btn_size - selector_h) * 0.5)
    self._draw_speed_selector(io, pal, shape)
    imgui.SameLine(0.0, 8.0)
    imgui.SetCursorPosY(row_y)
    if self._icon_button(
        ICON_RESET,
        'reset',
        pal,
        shape,
        keys=(('Backspace', 'Reset scene'),),
    ):
      self._reset_current_scene()

  def _draw_chrome(
      self,
      io: imgui.IO,
      pal: OverlayPalette,
      shape: OverlayShape,
      expanded: bool,
  ) -> None:
    """Draws the help button and the collapse/expand button."""
    del io  # Unused; kept so all row drawers share one signature.
    if self._icon_button(
        ICON_HELP,
        'help',
        pal,
        shape,
        keys=(('F1', 'Show or hide help'),),
        primary=self._show_help,
    ):
      self.toggle_help()
    imgui.SameLine(0.0, 4.0)
    collapse_icon = ICON_COLLAPSE if expanded else ICON_EXPAND
    if self._icon_button(collapse_icon, 'collapse', pal, shape):
      self.toggle_compact_card()

  def _draw_help_overlay(
      self,
      io: imgui.IO,
      pal: OverlayPalette,
      shape: OverlayShape,
      content_w: float,
  ) -> None:
    """Draws the centered MuJoCo 4.0.0 Interactive Release Notes overlay."""
    font0 = _get_font(io, 0)
    overlay_w = content_w + 2.0 * shape.pad_x
    max_h = max(240.0, float(io.DisplaySize.y) - 64.0)
    inner_w = content_w

    imgui.SetNextWindowPos(
        imgui.Vec2(io.DisplaySize.x * 0.5, io.DisplaySize.y * 0.5),
        imgui.Cond.Always,
        imgui.Vec2(0.5, 0.5),
    )
    imgui.SetNextWindowSizeConstraints(
        imgui.Vec2(overlay_w, 0.0), imgui.Vec2(overlay_w, max_h)
    )
    flags = (
        int(imgui.WindowFlags.NoTitleBar)
        | int(imgui.WindowFlags.NoResize)
        | int(imgui.WindowFlags.NoMove)
        | int(imgui.WindowFlags.NoCollapse)
        | int(imgui.WindowFlags.AlwaysAutoResize)
        | int(imgui.WindowFlags.NoSavedSettings)
        | int(imgui.WindowFlags.NoDocking)
        | int(imgui.WindowFlags.NoNav)
    )
    imgui.PushStyleVar(imgui.StyleVar.Alpha, 1.0)
    imgui.PushStyleVar(imgui.StyleVar.WindowRounding, shape.card_rounding)
    visible, _ = imgui.Begin('###MuJoCo4SlideshowHelp', None, flags)
    try:
      if not visible:
        return

      # Centered title.
      if font0 is not None:
        imgui.PushFont(font0, shape.title_px + 2.0)
      try:
        title_w = float(imgui.CalcTextSize(_RELEASE_NOTES_TITLE).x)
        imgui.SetCursorPosX(shape.pad_x + max(0.0, (inner_w - title_w) * 0.5))
        imgui.TextColored(pal.on_surface, _RELEASE_NOTES_TITLE)
      finally:
        if font0 is not None:
          imgui.PopFont()

      imgui.Spacing()
      if font0 is not None:
        imgui.PushFont(font0, shape.caption_px)
      try:
        imgui.PushStyleColor(imgui.Col.Text, pal.on_surface_var)
        imgui.PushTextWrapPos(shape.pad_x + inner_w)
        try:
          imgui.TextUnformatted(_RELEASE_NOTES_INTRO)
        finally:
          imgui.PopTextWrapPos()
          imgui.PopStyleColor()
      finally:
        if font0 is not None:
          imgui.PopFont()

      imgui.Spacing()
      imgui.Separator()
      imgui.Spacing()

      # Controls section.
      if font0 is not None:
        imgui.PushFont(font0, shape.caption_px + 1.0)
      try:
        imgui.TextColored(pal.on_surface, 'Controls')
      finally:
        if font0 is not None:
          imgui.PopFont()
      if font0 is not None:
        imgui.PushFont(font0, 13.0)
      try:
        if imgui.BeginTable('##SlideshowHelpControls', 2):
          for key_label, action in _SHORTCUT_ROWS:
            imgui.TableNextRow()
            imgui.TableNextColumn()
            imgui.TextColored(pal.accent_text, key_label)
            imgui.TableNextColumn()
            imgui.TextColored(pal.on_surface_var, f'  {action}')
          imgui.EndTable()
      finally:
        if font0 is not None:
          imgui.PopFont()

      imgui.Spacing()
      imgui.Separator()
      imgui.Spacing()

      # Demos section.
      if font0 is not None:
        imgui.PushFont(font0, shape.caption_px + 1.0)
      try:
        imgui.TextColored(pal.on_surface, 'Demos')
      finally:
        if font0 is not None:
          imgui.PopFont()
      if font0 is not None:
        imgui.PushFont(font0, 13.0)
      try:
        total_slides = len(SLIDES)
        for i, slide in enumerate(SLIDES):
          imgui.TextColored(self._dot_color(i, total_slides, pal), '•')
          imgui.SameLine(0.0, 0.0)
          imgui.TextColored(pal.on_surface_var, f'  {slide.title}')
      finally:
        if font0 is not None:
          imgui.PopFont()
    finally:
      imgui.End()
      imgui.PopStyleVar(2)

  def _draw_widgets_row(
      self,
      io: imgui.IO,
      pal: OverlayPalette,
      shape: OverlayShape,
      content_w: float,
      expanded: bool,
  ) -> None:
    """Draws the one-line control row, ``content_w`` wide.

    Play/pause, speed, reset, the timeline filling the remaining width, then
    help and collapse (or expand, when the card is collapsed to this row).

    Args:
      io: The ImGui IO.
      pal: The theme's colors.
      shape: The theme's geometry.
      content_w: The row's width.
      expanded: Whether the card shows its title and description.
    """
    row_y = float(imgui.GetCursorPosY())
    # One frame padding for the whole row, so that the slide counter shown on
    # narrow displays (a text item) is centered on the button-height line.
    pad_x = float(imgui.GetStyle().FramePadding.x)
    imgui.PushStyleVar(
        imgui.StyleVar.FramePadding,
        imgui.Vec2(pad_x, _row_frame_pad_y(shape)),
    )
    try:
      imgui.Dummy(imgui.Vec2(0.0, shape.btn_size))
      imgui.SameLine(0.0, 0.0)
      self._draw_controls(io, pal, shape, row_y)
      imgui.SameLine(0.0, shape.row_gap)
      label_w = self._time_label_width(io, shape)
      track_w = (
          content_w
          - self._controls_width(shape)
          - self._chrome_width(shape)
          - 2.0 * shape.row_gap
          - (label_w + 8.0)
      )
      self._draw_history_scrubber(io, pal, shape, max(80.0, track_w), row_y)
      imgui.SameLine(0.0, shape.row_gap)
      imgui.SetCursorPosY(row_y)
      self._draw_chrome(io, pal, shape, expanded)
    finally:
      imgui.PopStyleVar()

  def build_gui(self) -> None:
    """Builds the floating bottom-center caption card and control pill."""
    if self._status is None or imgui.GetCurrentContext() == 0:
      return

    io = imgui.GetIO()
    if io.DisplaySize.x <= 0 or io.DisplaySize.y <= 0:
      return

    now = float(imgui.GetTime())
    mouse_pos = io.MousePos
    if self._last_mouse_pos is None:
      self._last_mouse_pos = (float(mouse_pos.x), float(mouse_pos.y))
      mouse_moved = False
    else:
      dx = abs(float(mouse_pos.x) - self._last_mouse_pos[0])
      dy = abs(float(mouse_pos.y) - self._last_mouse_pos[1])
      mouse_moved = dx > 0.5 or dy > 0.5
      self._last_mouse_pos = (float(mouse_pos.x), float(mouse_pos.y))

    in_bottom_zone = 0.0 <= float(mouse_pos.x) <= float(
        io.DisplaySize.x
    ) and float(io.DisplaySize.y) - 160.0 <= float(mouse_pos.y) <= float(
        io.DisplaySize.y
    )
    if (mouse_moved and in_bottom_zone) or self._card_hovered_last_frame:
      self._last_mouse_bottom_t = now

    card_alpha = 1.0
    if self._card_state == CardState.HIDDEN:
      idle_s = now - self._last_mouse_bottom_t
      if idle_s >= 2.9:
        self._card_hovered_last_frame = False
        return
      if idle_s > 2.5:
        card_alpha = max(0.0, 1.0 - (idle_s - 2.5) / 0.4)

    self._apply_theme_if_needed()
    pal, shape = self._resolve_palette_and_shape()

    status = self._status
    show_caption = self._card_state == CardState.EXPANDED
    bottom_margin = 44.0 if self._has_studio_app else shape.bottom_margin
    win_pad = imgui.Vec2(shape.pad_x, shape.pad_y)
    if show_caption:
      win_rounding = shape.card_rounding
    else:
      # Collapsed to one row: a pill, rounded to half its height.
      pill_r = (shape.btn_size + 2.0 * shape.pad_y) * 0.5
      win_rounding = min(shape.card_rounding, pill_r)

    imgui.SetNextWindowPos(
        imgui.Vec2(io.DisplaySize.x * 0.5, io.DisplaySize.y - bottom_margin),
        imgui.Cond.Always,
        imgui.Vec2(0.5, 1.0),
    )

    window_flags = (
        int(imgui.WindowFlags.NoTitleBar)
        | int(imgui.WindowFlags.NoResize)
        | int(imgui.WindowFlags.NoMove)
        | int(imgui.WindowFlags.NoScrollbar)
        | int(imgui.WindowFlags.NoScrollWithMouse)
        | int(imgui.WindowFlags.NoCollapse)
        | int(imgui.WindowFlags.AlwaysAutoResize)
        | int(imgui.WindowFlags.NoSavedSettings)
        | int(imgui.WindowFlags.NoFocusOnAppearing)
        | int(imgui.WindowFlags.NoDocking)
        | int(imgui.WindowFlags.NoNav)
    )

    imgui.PushStyleVar(imgui.StyleVar.Alpha, card_alpha)
    imgui.PushStyleVar(imgui.StyleVar.WindowRounding, win_rounding)
    imgui.PushStyleVar(imgui.StyleVar.WindowBorderSize, 1.0)
    imgui.PushStyleVar(imgui.StyleVar.WindowPadding, win_pad)
    imgui.PushStyleVar(imgui.StyleVar.FrameRounding, shape.btn_rounding)
    imgui.PushStyleVar(imgui.StyleVar.ItemSpacing, imgui.Vec2(8.0, 6.0))
    imgui.PushStyleColor(imgui.Col.WindowBg, pal.surface)
    imgui.PushStyleColor(imgui.Col.PopupBg, pal.surface_hi)
    imgui.PushStyleColor(imgui.Col.Border, pal.outline)
    imgui.PushStyleColor(imgui.Col.Separator, pal.outline)
    imgui.PushStyleColor(imgui.Col.Text, pal.on_surface)
    imgui.PushStyleColor(imgui.Col.TextDisabled, pal.muted)
    imgui.PushStyleColor(imgui.Col.Button, _CLEAR)
    imgui.PushStyleColor(imgui.Col.ButtonHovered, pal.hover)
    imgui.PushStyleColor(imgui.Col.ButtonActive, pal.press)
    imgui.PushStyleColor(imgui.Col.Header, pal.selected)
    imgui.PushStyleColor(imgui.Col.HeaderHovered, pal.hover)

    try:
      content_w, caption_h = self._card_metrics(io, status, shape)
      visible, _ = imgui.Begin('###MuJoCo4SlideshowCard', None, window_flags)
      try:
        if visible:
          if show_caption:
            self._draw_header(io, status, pal, shape, content_w)
            imgui.Dummy(imgui.Vec2(0.0, 2.0))
            self._draw_caption(io, status, pal, shape, content_w, caption_h)
            imgui.Spacing()
            imgui.Separator()
            imgui.Spacing()
            self._draw_widgets_row(io, pal, shape, content_w, True)
          else:
            self._draw_widgets_row(io, pal, shape, content_w, False)
        self._card_hovered_last_frame = bool(
            imgui.IsWindowHovered(int(imgui.HoveredFlags.RootAndChildWindows))
        )
      finally:
        imgui.End()

      if self._show_help:
        self._draw_help_overlay(io, pal, shape, content_w)
    finally:
      imgui.PopStyleColor(11)
      imgui.PopStyleVar(6)

  @messages.handler
  def on_build_gui(self, _: messages.BuildGuiEvent) -> None:
    self.build_gui()
