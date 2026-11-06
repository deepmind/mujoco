---
name: mujoco-gui
description: >-
  Dear ImGui and ImPlot user interfaces across C++ and Python, immediate-mode
  widget design, layout and multi-row widget/column alignment (Tables API),
  tooltip length and aspect ratio, naming conventions (build_gui vs render), ID
  stack rules (## label disambiguation vs ### seed reset override), list
  virtualization (ImGuiListClipper), theming, input capture
  (WantCaptureKeyboard). Use for building, aligning, styling, and debugging Dear
  ImGui widgets and ImPlot charts. Do NOT use for Studio app
  architecture/plugins (mujoco-studio) or physics (mujoco-python).
---

# Dear ImGui Development & Architecture Guide

> [!TIP]
>
> **Official MuJoCo documentation (`doc/*.rst`) & related skills:**
>
> -   [`../../studio/python_plugins.rst`](../../studio/python_plugins.rst)
>     (`.. _StudioUiPlugins:`, `.. _StudioImGuiTips:`) and
>     [`../../studio/architecture.rst`](../../studio/architecture.rst)
>     (`.. _StudioCppComponents:`) — How Dear ImGui and ImPlot integrate into
>     MuJoCo Studio, `UpdateEvent` vs `BuildGuiEvent`, and C++ (`ux/`) vs Python
>     widget trade-offs. Read these short `.rst` chapters directly when present
>     locally or in `pip install mujoco` under
>     `pathlib.Path(mujoco.__file__).parent / 'doc'`, or fetch
>     `https://mujoco.readthedocs.io/en/latest/<path>.html` (or `/en/stable/`
>     once released; Sphinx anchors lowercase the label, e.g.
>     `#studioimguitips`).
> -   [`../../programming/ui.rst`](../../programming/ui.rst) (`.. _UI:`) —
>     Legacy `mjui` C framework used only in classic `simulate.cc` (distinct
>     from Studio's Dear ImGui stack).
> -   [mujoco-studio](../studio/SKILL.md) — Studio viewer launchers, decoupled
>     sim/viewer threads, plugins, hot reload, and `WebViewer` streaming.

> [!NOTE]
>
> Dear ImGui is an immediate-mode GUI framework designed for fast iterations,
> minimal state overhead, and high responsiveness. This guide documents best
> practices and conventions for writing maintainable, performant, and
> testable UI code across MuJoCo, its Python bindings, and dependent targets
> in C++, Python, and WebViewer.
>
> **Applies to both C++ and Python**: All architectural guidelines, widget
> patterns, Tables API usage, ID stack rules (`##` vs `###`), and performance
> principles apply equally whether authoring native C++ (`ImGui::`) or Python
> (`imgui.`) code. While snippets frequently illustrate patterns in Python for
> concise readability, the corresponding immediate-mode concepts, widget APIs,
> ID stack mechanics, clipper usage, and styling rules translate directly 1-to-1
> to C++. Guidelines apply both to core Dear ImGui and extension libraries, most
> importantly ImPlot.

## 1. Naming Conventions: `build_gui` vs `render` / `draw`

ImGui code does not render pixels to the screen directly. Instead, calling ImGui
functions builds and records **draw command lists** (vertex buffers, index
buffers, clip rects, and texture references) that are processed, rasterized,
and rendered later in the frame pipeline (via OpenGL, Vulkan, or
WebGL/NetImgui).

-   **Use `build_gui()` or `build_xxx_gui()`** (in Python) or `BuildGui()` /
    `BuildXxxGui()` (in C++) for all functions and methods that submit ImGui
    widgets (e.g., `build_control_panel_gui()`, `BuildControlPanelGui()`).
-   **Avoid `render()` or `draw()`** for UI layout functions. Reserve `render` /
    `draw` strictly for low-level graphics backend operations (such as
    `ImGui_ImplOpenGL3_RenderDrawData`).

```python
# Good: Accurately describes building the frame's UI hierarchy
def build_gui(self) -> None:
  self.build_header_gui()
  self.build_table_gui()

# Avoid: Suggests immediate rasterization/rendering
def draw_ui(self):
  ...
```

## 2. Decoupling GUI from Application Logic

GUI code must strictly handle user layout, presentation, and input forwarding.
The GUI is intended solely to glue the underlying logic together, and that glue
layer should be as small and thin as possible. Never mix business logic,
network/RPC queries, file operations, or simulation steps directly into frame
build functions.

-   **Minimal Glue Layer**: The GUI component should do little more than bind
    user inputs to underlying library calls and render current state.
-   **Separate Libraries**: Core business logic belongs in a standalone library
    or module (e.g., `inspector_lib.py` vs `inspector_gui.py`).
-   **Asynchronous Execution**: Expensive operations (RPCs, subprocesses, large
    searches) must run on worker threads/tasks, with GUI code reading cached
    state or progress flags.
-   **Testability**: Decoupled libraries can be thoroughly tested via standard
    unit tests without requiring an active ImGui context or window.

## 3. Custom Widgets: Immediate Mode Pure Functions

Custom widgets should remain lightweight, immediate-mode pure functions:

-   **Pure Immediate Mode**: Pass inputs as arguments, return output or mutation
    tuples (e.g., `changed, new_val = dial_widget(...)`). Avoid heavy stateful
    framework abstractions.
-   **Avoid Per-Widget Persistent State**: Storing state across frames inside a
    widget incurs a maintenance burden and limits reusability across arbitrary
    contexts. Keep widgets stateless whenever possible; if state is essential,
    store it in the parent application state rather than inside the widget.
-   **UX Customizations & Libraries**: Tailored visual controls (e.g., dial
    knobs, meters) are encouraged when they enhance UX. Instead of importing
    large external dependencies, copy and refactor small, focused snippets
    (must be MIT-licensed) into clean immediate-mode functions. In MuJoCo,
    shared custom ImGui widget functionality lives in the
    `imgui_widgets.h`/`cc` files in the Studio UX module.
-   **Test Engine Readiness**: Expose interaction endpoints and bounding boxes
    using Dear ImGui Test Engine hooks/macros so automated test drivers can
    interact with custom controls.
-   **Check Upstream & Changelog Before Implementing**: Before writing a new
    custom widget, layout behavior, or feature from scratch:
    -   Check the latest upstream changelog for the `docking` branch at
        https://github.com/ocornut/imgui/blob/docking/docs/CHANGELOG.txt and
        inspect what has changed relative to the vendored version in MuJoCo.
    -   Search Dear ImGui GitHub issues
        (https://github.com/ocornut/imgui/issues) and the Wiki
        (https://github.com/ocornut/imgui/wiki) for existing solutions, standard
        snippets, or discussions on the topic.
    -   **Check Local vs Upstream (Update Dependency)**: In general, check both
        the local checked-out code and the latest code on GitHub. If the
        feature, widget, or bugfix already exists in a newer upstream release,
        the cleaner solution is often to update the vendored dependency rather
        than maintaining custom workarounds in application code.

## 4. Disallowing Obsolete Functions & Internal Headers

ImGui maintains a stable public API and clearly documents deprecations:

-   **Ban Obsolete Functions**: Always define `IMGUI_DISABLE_OBSOLETE_FUNCTIONS`
    (and `IMGUI_DISABLE_OBSOLETE_KEYIO`) during builds and framework updates to
    prevent deprecated patterns from lingering in the codebase.
-   **Public API Stability**: Rely on the official public API (`imgui.h`). Avoid
    including internal headers (`imgui_internal.h`) in application and tool code
    unless implementing foundational backend extensions.

## 5. Performance & Virtualization: `ImGuiListClipper`

When rendering lists, tables, or log views that may contain many items (e.g.,
100+ bodies, simulation trajectories, log buffers):

-   **Always Virtualize with `ImGuiListClipper` / `imgui.ListClipper`**:
    Without clipping, ImGui generates vertices and text layouts for every item
    in the list even when offscreen. In remote streaming backends (NetImgui /
    WebViewer), unclipped tables saturate the WebSocket stream. In C++, use
    `ImGuiListClipper clipper; clipper.Begin(...)`; in Python, use
    `clipper = imgui.ListClipper(); clipper.Begin(...)`.
-   **Clipper Usage**: The clipper calculates the visible range and only submits
    the items currently in view.

```python
if imgui.BeginTable('ItemsTable', 1):
  clipper = imgui.ListClipper()
  clipper.Begin(len(items))
  while clipper.Step():
    for i in range(clipper.DisplayStart, clipper.DisplayEnd):
      imgui.TableNextRow()
      imgui.TableNextColumn()
      imgui.TextUnformatted(items[i].name)
  imgui.EndTable()
```

-   **Memory & Lifetime (`imgui.TextUnformatted`)**: Use `imgui.TextUnformatted`
    when displaying large or read-only text strings to avoid string copy and
    formatting overhead in Python bindings.

## 6. Runtime IDs, the ID Stack, and Test Automation (`##` vs `###`)

Dear ImGui identifies widgets by hashing their label text against the current ID
stack seed (`ImGuiWindow::GetID` calling `ImHashStr` in `imgui.cpp`).
Understanding the string operators `##` and `###` is critical for avoiding ID
collisions, maintaining state across dynamic label changes, and writing
automatable UIs.

### Technical Mechanics: `##` vs `###`

Both operators rely on Dear ImGui's internal text-rendering and hashing
routines:

-   **Display Boundary (`FindRenderedTextEnd`)**: Text rendering stops at the
    first occurrence of `##`. Because `###` also begins with `##`, **any
    characters following either `##` or `###` are hidden from the user**.
-   **Hidden Disambiguator (`##`)**:
    -   `ImHashStr` hashes the **entire string** (`"Label##Suffix"`), including
        both the visible label and the hidden suffix.
    -   **Purpose**: Disambiguating widgets that share the same visible label
        within the same ID scope (e.g., multiple "Delete" buttons or repeating
        row entries).
    -   **Why `Selectable(f'{body_name}##body_row_{i}')` uses `##`**: The
        visible label is `body_name`, but lists may contain duplicate body
        names or repeating entries. Appending `##body_row_{i}` hashes the full
        string `"{body_name}##body_row_{i}"`, producing a unique ID while
        keeping only `body_name` visible on screen. (Alternatively, wrap row
        rendering in `imgui.PushID(i)` / `imgui.PopID()`). If `###` were used
        here instead, `body_name` would be discarded completely from the hash.
-   **ID Override / Seed Reset (`###`)**:
    -   `ImHashStr` specifically detects three `#` characters (`###`). When
        encountered, it **resets the hash seed** back to the initial seed
        (`crc = seed`) and hashes **only** the substring following `###`. Any
        preceding visible text is completely discarded from the hash.
    -   **Dynamic Labels**: When visible text changes across frames (e.g.,
        `f'Items ({count})###ItemCounterBtn'` or animated frame titles), `###`
        guarantees that the widget's internal ID remains constant. This
        preserves active focus, hover state, window docking positions, and
        collapse states.
    -   **Semantically Meaningful Names for Automation**: For icon-only buttons
        (e.g., `f'{ICON_FA_PLAY}###PlayButton'`, where
        `ICON_FA_PLAY = '\uf04b'`), the icon glyph itself is constant across
        frames and would technically produce a stable hash even without `###`.
        However, without `###PlayButton`, the resulting ID is derived from an
        opaque unicode codepoint. Using `###PlayButton` assigns a
        human-readable, semantically meaningful identifier that facilitates
        agent-driven Dear ImGui Test Engine UI automation and computer control
        to reliably locate, inspect, and click the control by name
        (`"PlayButton"`).

> [!NOTE]
> Dear ImGui Test Engine is not currently bundled with MuJoCo, but support is
> planned, along with recorded UI interaction testing.

```python
# Icon glyph constant (e.g. FontAwesome play button unicode "\uf04b")
ICON_FA_PLAY = '\uf04b'

# Semantic ID override: The glyph is displayed, but the ID hashes only
# 'PlayButton'. Provides a meaningful identifier for test automation agents and
# tooling.
imgui.Button(f'{ICON_FA_PLAY}###PlayButton')

# Dynamic label with stable ID: Visible text changes with 'count', but the ID
# remains hashed from 'ItemCounterBtn', preserving focus, click, and hover
# state.
imgui.Button(f'Items ({count})###ItemCounterBtn')

# Disambiguated duplicate label: Visible text is 'body_name', but the ID hashes
# the full string '{body_name}##body_row_{i}' to prevent ID collisions across
# rows.
imgui.Selectable(f'{body_name}##body_row_{i}', is_selected)

# ID Scoping alternative: Using PushID / PopID for row loops instead of ##
# suffixes
imgui.PushID(i)
imgui.Selectable(body_name, is_selected)
imgui.PopID()
```

### Further ImGui & ImPlot References

Consult the upstream Dear ImGui and ImPlot sources and guides for authoritative
details and patterns:

-   **Dear ImGui ID Stack & API Guide**: `imgui.h` (around line 600) covers ID
    stack operations (`PushID`, `PopID`, `GetID`) and widget disambiguation
    rules.
-   **Dear ImGui Programmer Guide & String Hashing**: The top of `imgui.cpp`
    contains the comprehensive Programmer Guide; lines ~2470–2515 contain
    `ImHashStr` documenting `###` seed resetting.
-   **Dear ImGui FAQ**: The Dear ImGui [FAQ](https://github.com/ocornut/imgui/blob/master/docs/FAQ.md)
    includes dedicated sections on *"How can I have multiple widgets with the
    same label?"* and *"Why is my widget not reacting when I click on it?"*.
-   **Interactive ImGui Demo Window**: `imgui_demo.cpp` provides complete,
    working examples of every widget, table feature, and clipper pattern. Call
    `imgui.ShowDemoWindow()` in your application to explore features
    interactively.
-   **ImPlot Header & API Guide**: `implot.h` documents all plotting APIs,
    flags, styling properties, and context management.
-   **Interactive ImPlot Demo Window**: `implot_demo.cpp` contains extensive
    interactive demos for lines, scatter, bars, shaded plots, and realtime
    scrolling subplots. Call `implot.ShowDemoWindow()` to inspect
    implementations.

## 7. UI Design & Aesthetics: Grid Alignment (Tables API), Tooltips & Theming Costs

High-quality ImGui tools balance immediate-mode simplicity with clean visual
structure and consistent styling:

-   **Grid-Aligned Widgets & Form Rows (Prefer the Tables API)**:
    -   When arranging widgets into grids, key-value inspector panels, parameter
        sheets, or multi-row control bars where labels, dropdowns, or sliders
        must line up vertically and share equal widths, **prefer the modern
        Tables API** (`imgui.BeginTable`, `TableNextRow`, `TableNextColumn`).
    -   Avoid legacy or brittle alignment techniques such as `imgui.Columns()`
        (obsolete), chaining `SameLine()` after variable-length text labels
        (different label widths shift every subsequent control horizontally),
        hardcoding pixel widths with `SetNextItemWidth(px)`, or inserting dummy
        spacing.
    -   **Form & Multi-Control Alignment Recipe**:
        -   Pass `int(imgui.TableFlags.SizingStretchProp) | int(imgui.TableFlags.NoPadOuterX)`
            for borderless layout tables so the outer left and right edges stay
            flush with surrounding non-table widgets.
        -   Place text labels in `imgui.TableColumnFlags.WidthFixed` columns
            (omitting an explicit width so ImGui auto-fits to the widest label
            across all rows) and call `imgui.AlignTextToFramePadding()` before
            `TextUnformatted(...)` so label baselines center vertically against
            framed controls (`Combo`, `Slider`, `Input`).
        -   Place controls (`Combo`, `SliderInt`, `SliderFloat`, `InputFloat`)
            in `imgui.TableColumnFlags.WidthStretch` columns using hidden `##`
            labels and `imgui.SetNextItemWidth(-1)` so every control in the
            column starts at the exact same X coordinate and stretches to the
            exact same length. If sliders also have trailing right-hand labels
            of different lengths (e.g., `'Quality'` vs `'Overlay'`), place those
            trailing labels in their own `WidthFixed` column—calling
            `SetNextItemWidth(-1)` directly on a widget with a visible label
            sizes the frame box to the full column width and clips the label.

-   **Tooltip Length & Aspect Ratio**:
    -   Keep tooltips concise—avoid overly wordy descriptions that stretch into
        a wide single-line banner across the viewport.
    -   Because `SetTooltip` and `SetItemTooltip` do not wrap text
        automatically, insert explicit newlines (`\n`) to keep the tooltip box
        at a pleasing, compact aspect ratio (~25–40 characters per line across
        2–3 lines) rather than a long horizontal strip.
    -   Split lines at natural phrase, clause, or sentence boundaries (for
        example, between the primary action summary and secondary status or
        context) so each line reads cleanly on its own.

-   **Costs of Overriding Colors & Styles (Theming Impact)**:
    -   Dear ImGui manages color palettes and visual metrics globally
        (supporting dark mode, light mode, or custom studio themes). Semantic
        colors are queried via `ImGui::GetStyleColorVec4(...)` in C++ or
        `imgui.GetStyleColorVec4(...)` in Python.
    -   When custom widgets hardcode local color or style overrides (via
        `imgui.PushStyleColor` or `imgui.PushStyleVar`), **those values become
        decoupled from the global theme defaults**.
    -   If an application or viewer switches themes (such as toggling between
        dark and light modes) or updates global styling metrics, hardcoded local
        overrides will not adapt and cannot be edited or updated via global
        defaults, often causing unreadable contrast or visual inconsistencies.
    -   **Guideline**: Rely on semantic ImGui color tokens (`imgui.Col.Text`,
        `imgui.Col.FrameBg`, `imgui.Col.Button`, `imgui.Col.Header`, etc.) or
        derive dynamic colors via `imgui.GetStyleColorVec4(...)` rather than
        hardcoding static hex/RGB values. If a local override is strictly
        necessary, always pop it cleanly (`PopStyleColor` / `PopStyleVar`) and
        ensure it renders legibly across all supported themes.

```python
# Good: Multi-row control bar with vertically aligned labels and equal-length
# dropdowns/sliders using a borderless layout table
flags = int(imgui.TableFlags.SizingStretchProp) | int(imgui.TableFlags.NoPadOuterX)
if imgui.BeginTable('##controls_layout', 4, flags):
  imgui.TableSetupColumn('Label', int(imgui.TableColumnFlags.WidthFixed))
  imgui.TableSetupColumn('Combo', int(imgui.TableColumnFlags.WidthStretch), 1.1)
  imgui.TableSetupColumn('Slider', int(imgui.TableColumnFlags.WidthStretch), 0.9)
  imgui.TableSetupColumn('TrailingLabel', int(imgui.TableColumnFlags.WidthFixed))

  imgui.TableNextRow()
  imgui.TableNextColumn()
  imgui.AlignTextToFramePadding()
  imgui.TextUnformatted('Format:')
  imgui.TableNextColumn()
  imgui.SetNextItemWidth(-1)
  _, fmt_idx = imgui.Combo('##format', fmt_idx, ['Lossless', 'Lossy'])
  imgui.TableNextColumn()
  imgui.SetNextItemWidth(-1)
  _, quality = imgui.SliderInt('##quality', quality, 1, 100)
  imgui.TableNextColumn()
  imgui.AlignTextToFramePadding()
  imgui.TextUnformatted('Quality')

  imgui.TableNextRow()
  imgui.TableNextColumn()
  imgui.AlignTextToFramePadding()
  imgui.TextUnformatted('Viewport:')
  imgui.TableNextColumn()
  imgui.SetNextItemWidth(-1)
  _, cam_idx = imgui.Combo('##camera', cam_idx, camera_names)
  imgui.TableNextColumn()
  imgui.SetNextItemWidth(-1)
  _, opacity = imgui.SliderFloat('##overlay', opacity, 0.0, 1.0)
  imgui.TableNextColumn()
  imgui.AlignTextToFramePadding()
  imgui.TextUnformatted('Overlay')

  imgui.EndTable()

# Avoid: Hardcoding colors detaches widgets from global themes
# (breaks dark/light modes)
# imgui.PushStyleColor(imgui.Col.Text, (0.1, 0.1, 0.1, 1.0))  # Unreadable in dark mode!

# Good: Rely on semantic theme colors from ImGui
text_color = imgui.GetStyleColorVec4(imgui.Col.Text)
```

## 8. Searchability of UI Strings in Source Code

Developers and agents often search the codebase for text seen on screen to
locate the corresponding feature implementation:

-   **Single-Line Static Strings**: Keep user-visible label strings, button
    names, and headings on a single source line rather than breaking phrases
    across lines.
-   **Conservative Interpolation**: Avoid over-fragmenting strings or splitting
    prefixes/suffixes across multiple format variables when a single cohesive
    string or format template makes the UI text searchable.

## 9. Python Bindings

-   **Thin Binding Layer (No GUI/Application Logic in Bindings)**: Just as GUI
    code should avoid containing heavy application logic in glue layers,
    `pybind11` bindings (e.g., `ux.cc`) must remain thin wrappers around C++
    functions (such as `ux/*.h` and `ux/*.cc`). Do not write multi-widget ImGui
    layout or application logic directly inside Python binding lambdas;
    implement the widget or helper in C++ and bind the function instead.
-   **Add On-Demand (Never Remove)**: Expose new ImGui or ImPlot bindings to
    Python only when they are actually needed and used by an application or
    plugin. Once added, do not remove them.
-   **Order Parity**: When exposing new functions, add them in the exact order
    they appear in the upstream C++ headers to simplify future version upgrades
    and diffs.
-   **Keyboard Capture Guarding (`io.WantCaptureKeyboard`)**: When implementing
    keyboard shortcuts (such as Space for pause or Backspace for reset), always
    check `not imgui.GetIO().WantCaptureKeyboard` before handling the key event.
    When a text input, filter box, or popup is active, ImGui sets
    `WantCaptureKeyboard = True`; ignoring this flag causes keystrokes typed
    into an input field to trigger simulation actions.
-   **Initial Focus in Search & Popups (`SetKeyboardFocusHere`)**: When opening
    a command palette, filter box, or modal dialog, call
    `imgui.SetKeyboardFocusHere()` immediately before the `imgui.InputText` or
    `InputTextWithHint` call so the input field receives focus immediately
    without requiring a manual click.
