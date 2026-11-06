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

import math
import sys
import threading
import time

from absl.testing import absltest
import mujoco
from mujoco.sample.studio.overview import slideshow_plugins
from mujoco.studio import launch_thread
from mujoco.studio import messages
from mujoco.studio import plugin
from mujoco.studio import sim
from mujoco.studio import sim_history
from mujoco.studio import step_control
from mujoco.studio import ux
from mujoco.studio import viewer_handle
import numpy as np


class _FakeViewer:
  """The parts of a viewer that SlideshowOverlay's model handler touches."""

  def __init__(self, sent: list[messages.Event | messages.Snapshot]):
    self.perturb = mujoco.MjvPerturb()
    self.camera = mujoco.MjvCamera()
    self.vis_options = mujoco.MjvOption()
    self._sent = sent

  def send_to_sim(self, message: messages.Event | messages.Snapshot) -> None:
    self._sent.append(message)


class SlideshowTest(absltest.TestCase):

  def setUp(self):
    super().setUp()
    ux.init_console()
    ux.clear_console()

    sys_path = list(sys.path)
    modules = dict(sys.modules)

    def restore():
      sys.path[:] = sys_path
      for name in set(sys.modules) - set(modules):
        del sys.modules[name]
      sys.modules.update(modules)

    self.addCleanup(restore)

  def test_load_plugins_discovers_slideshow_plugins(self):
    loaded = plugin.load_plugins([slideshow_plugins.__file__])
    self.assertEqual(
        [type(p).__name__ for p in loaded],
        ['SlideshowSim', 'SlideshowOverlay'],
    )
    viewer_plugins, sim_plugins = plugin.split_plugins(loaded)
    self.assertLen(viewer_plugins, 1)
    self.assertLen(sim_plugins, 1)

  def test_resolve_slide_index(self):
    self.assertEqual(slideshow_plugins.resolve_slide_index('1'), 0)
    self.assertEqual(
        slideshow_plugins.resolve_slide_index(
            str(len(slideshow_plugins.SLIDES))
        ),
        len(slideshow_plugins.SLIDES) - 1,
    )
    self.assertEqual(slideshow_plugins.resolve_slide_index('ipc'), 1)
    self.assertGreater(slideshow_plugins.resolve_slide_index('newton'), 0)
    self.assertGreater(slideshow_plugins.resolve_slide_index('chain'), 0)
    self.assertEqual(
        slideshow_plugins.resolve_slide_index('SCALING'),
        len(slideshow_plugins.SLIDES) - 1,
    )
    with self.assertRaises(ValueError):
      slideshow_plugins.resolve_slide_index('0')
    with self.assertRaises(ValueError):
      slideshow_plugins.resolve_slide_index('nonexistent_slug')

  def test_slides_are_grouped_by_category(self):
    seen_categories: list[str] = []
    for slide in slideshow_plugins.SLIDES:
      if not seen_categories or seen_categories[-1] != slide.category:
        self.assertNotIn(
            slide.category,
            seen_categories,
            msg=f'Category {slide.category!r} appears in non-adjacent blocks.',
        )
        seen_categories.append(slide.category)

  def test_all_slides_compile_and_step_cleanly(self):
    sim_plugin = slideshow_plugins.SlideshowSim()
    self.assertLen(slideshow_plugins.SLIDES, 18)

    for idx, slide in enumerate(slideshow_plugins.SLIDES):
      with self.subTest(slide=slide.slug):
        ux.clear_console()
        model, path = sim_plugin.compile_slide(idx)
        self.assertIsNotNone(model)
        if slide.model_relpath:
          self.assertTrue(path.endswith(slide.model_relpath))
        else:
          self.assertEqual(path, '')
        if slide.slug == 'ccd':
          self.assertAlmostEqual(float(model.vis.global_.realtime), 0.2)

        data = mujoco.MjData(model)
        if slide.init_fn is not None:
          slide.init_fn(model, data)
        mujoco.mj_forward(model, data)

        camera = mujoco.MjvCamera()
        slideshow_plugins.apply_interactive_camera(
            model, data, camera, slide.camera
        )
        self.assertEqual(camera.type, int(mujoco.mjtCamera.mjCAMERA_FREE))
        self.assertEqual(camera.fixedcamid, -1)
        self.assertGreater(camera.distance, 0.2)
        self.assertTrue(np.all(np.isfinite(camera.lookat)))

        for _ in range(15):
          if slide.pre_step_fn is not None:
            slide.pre_step_fn(model, data)
          mujoco.mj_step(model, data)
        self.assertTrue(np.all(np.isfinite(data.qpos)))

        counts = ux.get_console_counts()
        self.assertEqual(
            counts.errors,
            0,
            msg=f'Console errors on {slide.slug}: {ux.get_console_lines()}',
        )

  def test_clay_press_stamps_and_retracts(self):
    sim_plugin = slideshow_plugins.SlideshowSim()
    clay_idx = slideshow_plugins.resolve_slide_index('clay')
    slide = slideshow_plugins.SLIDES[clay_idx]
    model, _ = sim_plugin.compile_slide(clay_idx)
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)

    head_id = mujoco.mj_name2id(model, int(mujoco.mjtObj.mjOBJ_BODY), 'head')
    self.assertGreater(head_id, 0)
    z_initial = float(data.xpos[head_id, 2])
    z_stamped = z_initial

    while data.time < 0.75:
      if slide.pre_step_fn is not None:
        slide.pre_step_fn(model, data)
      mujoco.mj_step(model, data)
      z_stamped = min(z_stamped, float(data.xpos[head_id, 2]))
    self.assertLess(z_stamped, z_initial - 0.12)

    while data.time < 2.0:
      if slide.pre_step_fn is not None:
        slide.pre_step_fn(model, data)
      mujoco.mj_step(model, data)
    z_retracted = float(data.xpos[head_id, 2])
    self.assertGreater(z_retracted, z_stamped + 0.10)

  def test_froude_scaling_ratios(self):
    model = slideshow_plugins._build_froude_scaling_model()
    half_mount = mujoco.mj_name2id(
        model, int(mujoco.mjtObj.mjOBJ_BODY), 'half_upper'
    )
    unit_mount = mujoco.mj_name2id(
        model, int(mujoco.mjtObj.mjOBJ_BODY), 'unit_upper'
    )
    large_mount = mujoco.mj_name2id(
        model, int(mujoco.mjtObj.mjOBJ_BODY), 'large_upper'
    )
    self.assertAlmostEqual(
        float(model.body_mass[half_mount]) / float(model.body_mass[unit_mount]),
        0.5**3,
        places=5,
    )
    self.assertAlmostEqual(
        float(model.body_mass[large_mount])
        / float(model.body_mass[unit_mount]),
        1.8**3,
        places=4,
    )
    data = mujoco.MjData(model)
    slideshow_plugins._init_froude_scaling(model, data)
    self.assertAlmostEqual(float(data.qpos[0]), 1.15)
    self.assertAlmostEqual(float(data.qpos[2]), 1.15)
    self.assertAlmostEqual(float(data.qpos[4]), 1.15)

  def test_draggable_bodies_hang_off_joints(self):
    model = slideshow_plugins._build_froude_scaling_model()
    body = lambda name: mujoco.mj_name2id(
        model, int(mujoco.mjtObj.mjOBJ_BODY), name
    )
    # The world (and so the floor and gantry) and the jointless pivot mounts
    # orbit the camera; the pendulum links and their children can be pulled.
    self.assertFalse(slideshow_plugins.is_draggable_body(model, 0))
    self.assertFalse(
        slideshow_plugins.is_draggable_body(model, body('unit_mount'))
    )
    self.assertTrue(
        slideshow_plugins.is_draggable_body(model, body('unit_upper'))
    )
    self.assertTrue(
        slideshow_plugins.is_draggable_body(model, body('large_lower'))
    )
    self.assertFalse(slideshow_plugins.is_draggable_body(model, -1))
    self.assertFalse(slideshow_plugins.is_draggable_body(model, model.nbody))

  def test_slide_navigation_over_endpoints(self):
    viewer_ep, sim_ep = launch_thread.make_thread_endpoints()
    sim_plugin = slideshow_plugins.SlideshowSim(
        initial_slide=0, prefetch=False
    )
    handle = viewer_handle.ViewerHandle(sim_ep, sim_plugins=[sim_plugin])
    self.addCleanup(handle.close)

    # Initial slide 0 ('pbr') is published on SimInitEvent.
    initial_events = viewer_ep.get_sim_events()
    slide_events = [
        e
        for e in initial_events
        if isinstance(e, slideshow_plugins.SlideChangedEvent)
    ]
    model_events = [
        e for e in initial_events if isinstance(e, messages.ModelEvent)
    ]
    self.assertLen(slide_events, 1)
    self.assertLen(model_events, 1)
    self.assertEqual(slide_events[0].slug, 'pbr')
    self.assertEqual(slide_events[0].index, 0)

    # Advance forward (+1) and wrap backward (-2 -> last slide).
    viewer_ep.send_to_sim(slideshow_plugins.SelectSlideEvent(delta=1))
    handle.sync()
    events = [
        e
        for e in viewer_ep.get_sim_events()
        if isinstance(e, slideshow_plugins.SlideChangedEvent)
    ]
    self.assertLen(events, 1)
    self.assertEqual(events[0].slug, slideshow_plugins.SLIDES[1].slug)

    viewer_ep.send_to_sim(slideshow_plugins.SelectSlideEvent(delta=-2))
    handle.sync()
    events = [
        e
        for e in viewer_ep.get_sim_events()
        if isinstance(e, slideshow_plugins.SlideChangedEvent)
    ]
    self.assertLen(events, 1)
    self.assertEqual(events[0].index, len(slideshow_plugins.SLIDES) - 1)
    self.assertEqual(events[0].slug, 'scaling')

    # Reset reapplies initial angles on the scaling slide.
    assert handle.data is not None
    handle.data.qpos[:] = 0.0
    viewer_ep.send_to_sim(messages.ResetEvent())
    handle.sync()
    self.assertTrue(math.isclose(float(handle.data.qpos[0]), 1.15))

  def test_deck_compile_order_is_nearest_first(self):
    self.assertEqual(
        slideshow_plugins.deck_compile_order(0, 6), [1, 5, 2, 4, 3]
    )
    self.assertEqual(
        slideshow_plugins.deck_compile_order(4, 6), [5, 3, 0, 2, 1]
    )
    self.assertEqual(slideshow_plugins.deck_compile_order(0, 1), [])

  def test_deck_is_precompiled_and_announced_nearest_first(self):
    viewer_ep, sim_ep = launch_thread.make_thread_endpoints()
    sim_plugin = slideshow_plugins.SlideshowSim(initial_slide=2)
    handle = viewer_handle.ViewerHandle(sim_ep, sim_plugins=[sim_plugin])
    self.addCleanup(handle.close)
    viewer_ep.get_sim_events()  # The initial slide's events.

    # The other slides arrive as PreloadModelEvents over the next sync()
    # calls, nearest slides first, each carrying the compiled model the deck
    # keeps for that slide.
    total = len(slideshow_plugins.SLIDES)
    hints: list[messages.PreloadModelEvent] = []
    deadline = time.monotonic() + 120.0
    while len(hints) < total - 1 and time.monotonic() < deadline:
      handle.sync()
      hints += [
          e
          for e in viewer_ep.get_sim_events()
          if isinstance(e, messages.PreloadModelEvent)
      ]
      time.sleep(0.01)
    self.assertLen(hints, total - 1)
    order = slideshow_plugins.deck_compile_order(2, total)
    deck = sim_plugin._deck  # pylint: disable=protected-access
    for hint, i in zip(hints, order):
      self.assertIs(hint.model, deck[i][0])

    # Switching to a compiled slide publishes that very model, with fresh data
    # and the slide's initial state, instead of compiling again.
    scaling = slideshow_plugins.resolve_slide_index('scaling')
    scaling_hint = hints[order.index(scaling)]
    viewer_ep.send_to_sim(slideshow_plugins.SelectSlideEvent(index=scaling))
    handle.sync()
    model_events = [
        e
        for e in viewer_ep.get_sim_events()
        if isinstance(e, messages.ModelEvent)
    ]
    self.assertLen(model_events, 1)
    self.assertIs(model_events[0].model, scaling_hint.model)
    assert handle.data is not None
    self.assertTrue(math.isclose(float(handle.data.qpos[0]), 1.15))

  def test_prefetch_off_compiles_on_demand_without_hints(self):
    viewer_ep, sim_ep = launch_thread.make_thread_endpoints()
    sim_plugin = slideshow_plugins.SlideshowSim(
        initial_slide=0, prefetch=False
    )
    handle = viewer_handle.ViewerHandle(sim_ep, sim_plugins=[sim_plugin])
    self.addCleanup(handle.close)
    for _ in range(5):
      handle.sync()
      time.sleep(0.01)
    self.assertEmpty([
        e
        for e in viewer_ep.get_sim_events()
        if isinstance(e, messages.PreloadModelEvent)
    ])

  def test_close_stops_and_joins_deck_compile_thread(self):
    tiny_xml = (
        '<mujoco><worldbody><body><freejoint/><geom size="0.1"/></body>'
        '</worldbody></mujoco>'
    )
    slide_1_started = threading.Event()
    slide_2_compiled = False
    sim_plugin: slideshow_plugins.SlideshowSim | None = None

    def build_slide_1() -> mujoco.MjModel:
      slide_1_started.set()
      self.assertIsNotNone(sim_plugin)
      # pylint: disable-next=protected-access
      cancel = sim_plugin._compile_cancel if sim_plugin is not None else None
      self.assertIsNotNone(cancel)
      if cancel is not None:
        self.assertTrue(cancel.wait(timeout=5.0))
      return mujoco.MjModel.from_xml_string(tiny_xml)

    def build_slide_2() -> mujoco.MjModel:
      nonlocal slide_2_compiled
      slide_2_compiled = True
      return mujoco.MjModel.from_xml_string(tiny_xml)

    slides = (
        slideshow_plugins.SlideSpec(
            slug='s0',
            category='Test',
            title='S0',
            caption='',
            build_model_fn=lambda: mujoco.MjModel.from_xml_string(tiny_xml),
        ),
        slideshow_plugins.SlideSpec(
            slug='s1',
            category='Test',
            title='S1',
            caption='',
            build_model_fn=build_slide_1,
        ),
        slideshow_plugins.SlideSpec(
            slug='s2',
            category='Test',
            title='S2',
            caption='',
            build_model_fn=build_slide_2,
        ),
    )
    _, sim_ep = launch_thread.make_thread_endpoints()
    sim_plugin = slideshow_plugins.SlideshowSim(
        slides=slides, initial_slide=0, prefetch=True
    )
    handle = viewer_handle.ViewerHandle(sim_ep, sim_plugins=[sim_plugin])
    self.addCleanup(handle.close)

    self.assertTrue(slide_1_started.wait(timeout=5.0))
    thread = sim_plugin._compile_thread  # pylint: disable=protected-access
    self.assertIsNotNone(thread)
    if thread is not None:
      self.assertTrue(thread.is_alive())

    handle.close()
    if thread is not None:
      self.assertFalse(thread.is_alive())
    # pylint: disable-next=protected-access
    self.assertIsNone(sim_plugin._compile_thread)
    self.assertFalse(slide_2_compiled)
    # pylint: disable-next=protected-access
    self.assertTrue(sim_plugin._compiled.empty())

  def test_overlay_follows_the_viewer_theme(self):
    overlay = slideshow_plugins.SlideshowOverlay()
    self.assertEqual(overlay.theme, ux.GuiTheme.DARK)
    self.assertTrue(overlay.is_dark)

    # Material shape and the Gemini gradient ramp on every theme.
    for spec in slideshow_plugins.THEMES.values():
      self.assertEqual(spec.shape.card_rounding, 22.0)
      self.assertEqual(spec.shape.btn_rounding, 16.0)
      self.assertEqual(spec.shape.btn_size, 32.0)
      self.assertLen(spec.palette.dot_ramp, 3)

    # The card switches with the viewer's theme (the Studio app's theme menu or
    # the web viewer's role bar); the classic theme gets the dark card.
    overlay.on_theme(plugin.ThemeEvent(theme=ux.GuiTheme.LIGHT))
    self.assertEqual(overlay.theme, ux.GuiTheme.LIGHT)
    self.assertFalse(overlay.is_dark)
    overlay.on_theme(plugin.ThemeEvent(theme=ux.GuiTheme.CLASSIC))
    self.assertEqual(overlay.theme, ux.GuiTheme.CLASSIC)
    self.assertTrue(overlay.is_dark)
    self.assertIs(
        slideshow_plugins.THEMES[ux.GuiTheme.CLASSIC],
        slideshow_plugins.THEMES[ux.GuiTheme.DARK],
    )

  def test_format_sim_time(self):
    self.assertEqual(slideshow_plugins._format_sim_time(0.0), '0.00 s')
    self.assertEqual(slideshow_plugins._format_sim_time(1.234), '1.23 s')
    self.assertEqual(slideshow_plugins._format_sim_time(12.34), '12.3 s')
    self.assertEqual(slideshow_plugins._format_sim_time(123.4), '123 s')
    self.assertEqual(slideshow_plugins._format_sim_time(-1.0), '0.00 s')

  def test_history_scrub_rewinds_and_pauses_the_sim(self):
    viewer_ep, sim_ep = launch_thread.make_thread_endpoints()
    handle = viewer_handle.ViewerHandle(
        sim_ep,
        sim_plugins=[
            step_control.StepControl(),
            sim_history.SimHistory(),
            slideshow_plugins.SlideshowSim(initial_slide=0, prefetch=False),
        ],
    )
    self.addCleanup(handle.close)

    # Let the sim record some history, then rewind ten frames the way the
    # overlay's slider does: a SimHistorySnapshot with the desired index.
    deadline = time.monotonic() + 30.0
    history = None
    while time.monotonic() < deadline:
      handle.sync()
      viewer_ep.get_sim_events()
      for snapshot in viewer_ep.get_sim_snapshots():
        if isinstance(snapshot, sim_history.SimHistorySnapshot):
          history = snapshot
      if history is not None and history.size > 20:
        break
    self.assertIsNotNone(history)
    assert history is not None
    self.assertEqual(history.index, 0)
    head_time = history.head_time

    viewer_ep.send_to_sim(sim_history.SimHistorySnapshot(index=-10))
    handle.sync()
    assert handle.data is not None
    rewound = None
    paused = None
    for snapshot in viewer_ep.get_sim_snapshots():
      if isinstance(snapshot, sim_history.SimHistorySnapshot):
        rewound = snapshot
      elif isinstance(snapshot, messages.StepControlSnapshot):
        paused = snapshot
    self.assertIsNotNone(rewound)
    assert rewound is not None
    self.assertEqual(rewound.index, -10)
    self.assertLess(float(handle.data.time), head_time)
    self.assertIsNotNone(paused)
    assert paused is not None
    self.assertEqual(paused.pause_state, sim.PauseState.NORMAL_PAUSED)

  def test_sim_step_control_uses_scene_authored_realtime_speed(self):
    ccd_index = slideshow_plugins.resolve_slide_index('ccd')
    viewer_ep, sim_ep = launch_thread.make_thread_endpoints()
    sim_stepper = step_control.StepControl()
    handle = viewer_handle.ViewerHandle(
        sim_ep,
        sim_plugins=[
            sim_stepper,
            slideshow_plugins.SlideshowSim(
                initial_slide=ccd_index, prefetch=False
            ),
        ],
    )
    self.addCleanup(handle.close)
    self.assertAlmostEqual(sim_stepper.step_control.get_speed(), 20.0)

    viewer_ep.send_to_sim(slideshow_plugins.SelectSlideEvent(index=0))
    handle.sync()
    self.assertAlmostEqual(sim_stepper.step_control.get_speed(), 100.0)

  def test_overlay_step_control_restarts_with_each_new_scene(self):
    # Each new scene starts running at its authored <global realtime=.../>
    # speed; reloading the same file preserves user pause/speed settings.
    overlay = slideshow_plugins.SlideshowOverlay()
    sent: list[messages.Event | messages.Snapshot] = []
    overlay.viewer = _FakeViewer(sent)
    unit_model = mujoco.MjModel.from_xml_string(
        '<mujoco><worldbody><body><freejoint/><geom size="0.1"/></body>'
        '</worldbody></mujoco>'
    )
    unit_data = mujoco.MjData(unit_model)
    mujoco.mj_forward(unit_model, unit_data)
    slow_model = mujoco.MjModel.from_xml_string(
        '<mujoco><visual><global realtime="0.2"/></visual>'
        '<worldbody><body><freejoint/><geom size="0.1"/></body></worldbody>'
        '</mujoco>'
    )
    slow_data = mujoco.MjData(slow_model)
    mujoco.mj_forward(slow_model, slow_data)

    def load(
        model: mujoco.MjModel, data: mujoco.MjData, path: str
    ) -> list[messages.StepControlSnapshot]:
      sent.clear()
      overlay.on_post_model(
          messages.PostModelEvent(model=model, data=data, path=path)
      )
      return [s for s in sent if isinstance(s, messages.StepControlSnapshot)]

    (snapshot,) = load(unit_model, unit_data, '/scenes/a.xml')
    self.assertEqual(snapshot.pause_state, sim.PauseState.UNPAUSED)
    self.assertAlmostEqual(snapshot.speed, 100.0)

    # The user pauses and slows down; reloading the same file keeps both, as
    # the sim does, so nothing needs announcing.
    overlay.on_step_control(
        messages.StepControlSnapshot(
            pause_state=sim.PauseState.NORMAL_PAUSED,
            speed=40.0,
            noise_scale=0.0,
            noise_rate=0.0,
        )
    )
    self.assertEmpty(load(unit_model, unit_data, '/scenes/a.xml'))

    # A new scene with <global realtime="0.2"/> starts running at 20%, and a
    # subsequent default-speed scene returns to 100%.
    (snapshot,) = load(slow_model, slow_data, '/scenes/slow.xml')
    self.assertEqual(snapshot.pause_state, sim.PauseState.UNPAUSED)
    self.assertAlmostEqual(snapshot.speed, 20.0)

    (snapshot,) = load(unit_model, unit_data, '/scenes/b.xml')
    self.assertEqual(snapshot.pause_state, sim.PauseState.UNPAUSED)
    self.assertAlmostEqual(snapshot.speed, 100.0)

  def test_overlay_card_state_transitions(self):
    overlay = slideshow_plugins.SlideshowOverlay(hide_card=False)
    self.assertEqual(overlay.card_state, slideshow_plugins.CardState.EXPANDED)
    self.assertFalse(overlay.hide_card)

    # H cycles EXPANDED -> COMPACT -> HIDDEN -> EXPANDED.
    self.assertEqual(
        overlay.cycle_card_state(), slideshow_plugins.CardState.COMPACT
    )
    self.assertFalse(overlay.hide_card)
    self.assertEqual(
        overlay.cycle_card_state(), slideshow_plugins.CardState.HIDDEN
    )
    self.assertTrue(overlay.hide_card)
    self.assertEqual(
        overlay.cycle_card_state(), slideshow_plugins.CardState.EXPANDED
    )
    self.assertFalse(overlay.hide_card)

    # Chevron button toggles EXPANDED <-> COMPACT (and expands from HIDDEN).
    self.assertEqual(
        overlay.toggle_compact_card(), slideshow_plugins.CardState.COMPACT
    )
    self.assertEqual(
        overlay.toggle_compact_card(), slideshow_plugins.CardState.EXPANDED
    )
    hidden_overlay = slideshow_plugins.SlideshowOverlay(hide_card=True)
    self.assertEqual(
        hidden_overlay.card_state, slideshow_plugins.CardState.HIDDEN
    )
    self.assertTrue(hidden_overlay.hide_card)
    self.assertEqual(
        hidden_overlay.toggle_compact_card(),
        slideshow_plugins.CardState.EXPANDED,
    )


if __name__ == '__main__':
  absltest.main()
