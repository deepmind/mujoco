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
"""Python web server for the MuJoCo Web Viewer.

This server runs on a background daemon thread with one asyncio loop on a single
public port:

  * Plain HTTP GET   serves static files (index.html, WASM, assets), /model.
  * WebSocket /ui    serves the bridge to the headless NetImgui client, which
                     connects over loopback TCP (see headless_ui.cc).
  * WebSocket /state serves the latest-wins state payload broadcast at ~60Hz
                     (see state_payload.h) as binary frames, and session roster
                     updates as text frames.
  * WebSocket /drop  receives models dragged onto the page (one binary frame per
                     file, controller only; see drop_handler).

Because everything is served through one port, one firewall rule, port-forward,
or HTTPS tunnel exposes the whole viewer. The browser derives its WebSocket URLs
from the page origin so no client configuration is needed.

One browser at a time controls the interactive session (it owns the /ui bridge);
later browsers are rejected from /ui with WebSocket close code
_WS_CLOSE_CONTROLLER_TAKEN and can take the controller slot once it frees up.
/state connections beyond the spectator limit are closed with close code
_WS_CLOSE_SESSION_FULL.
"""

import asyncio
import datetime
import enum
import functools
import json
import logging
import os
import queue
import socket
import struct
import sys
import threading
from typing import Any, Awaitable, Callable, cast
import urllib.parse

from websockets.asyncio.server import serve
from websockets.asyncio.server import ServerConnection
from websockets.datastructures import Headers
from websockets.exceptions import ConnectionClosed
from websockets.exceptions import InvalidHandshake
from websockets.frames import CloseCode
from websockets.http11 import Request
from websockets.http11 import Response

# Our custom WebSocket close codes in range 4xxx. The client shows a notice
# and retries slowly (see web_client_session.cc / web_client.cc).
_WS_CLOSE_CONTROLLER_TAKEN = 4001  # /ui:    another browser is controlling.
_WS_CLOSE_SESSION_FULL = 4002  # /state: the spectator limit is reached.
_WS_CLOSE_INACTIVE = 4003  # /state: hidden tab kicked to free a slot.
_WS_CLOSE_NOT_CONTROLLER = 4004  # /drop: only the controller may load models.
_WS_CLOSE_VIEWER_ERROR = 4005  # /state: the viewer stopped with an error.

# A WebSocket close frame carries at most 123 bytes of reason text.
_WS_CLOSE_REASON_MAX_BYTES = 123

# Upper bound on the runtime-editable spectator limit.
_MAX_SPECTATOR_HARD_CAP = 32

# How many browsers may watch in addition to the controller; the controller can
# change it at runtime (up to _MAX_SPECTATOR_HARD_CAP). The limit is about
# session behaviour — keeping the crowd and the control-request queue at a
# manageable size — not about resources: a state payload is a few KB at
# 60Hz, serialized once and sent per viewer, so even a full session costs
# only a few Mbit/s of upload. There is deliberately no config or CLI
# option (spectating exposes nothing that controlling does not).
_MAX_SPECTATORS_DEFAULT = 8

# Controller-only message prefix carrying the new spectator limit.
_MAX_SPECTATORS_PREFIX = "max_spectators="
# Sent to the page whose control claim the controller slot is reserved for.
_GRANT_MESSAGE = "grant"
# Sent by the browser after applying each state payload (keep in sync:
# web_client_session.cc). Flow control for the /state stream: at most one
# payload is in flight per client, so a slow link carries the freshest state
# it can instead of buffering seconds of stale payloads in the socket.
_STATE_ACK_MESSAGE = "state_ack"

# A lost ack must not stall the stream forever; after this long the next
# payload is sent unacked.
_STATE_ACK_TIMEOUT_SEC = 2.0

# A control grant must arrive within this window, else it moves down the queue.
_GRANT_EXPIRY_SEC = 5.0


# A controller whose tab stops running (hidden or closed without a clean
# disconnect) is released once someone is waiting for control. Detected by
# silence: a live controller tab sends input packets every frame.
_CONTROLLER_SILENT_KICK_SEC = 90.0

# Spectators are kicked after this much heartbeat silence (their tab is hidden
# or gone), freeing a viewer slot. Live tabs heartbeat every ~30s.
_SPECTATOR_SILENT_KICK_SEC = 300.0

# Default public port, and how many consecutive ports to try when it is taken
# (e.g. by another running viewer).
_DEFAULT_HTTP_PORT = 8080
_PORT_SCAN_COUNT = 20

# NetImgui's CmdVersion handshake packet is always 120 bytes.
_NETIMGUI_CMD_VERSION_SIZE = 120

# How long a browser holding the controller slot waits for the headless NetImgui
# client to (re)connect over loopback before the slot is released. The client
# reconnects within ~1s in the normal case; this only bounds the pathological
# one where it never appears (e.g., the headless UI failed to start), so a stuck
# controller cannot lock every other browser out forever.
_UI_TCP_WAIT_SEC = 30.0

# Content types for the static files the HTTP handler serves.
_CONTENT_TYPES = {
    ".html": "text/html; charset=utf-8",
    ".js": "text/javascript",
    ".wasm": "application/wasm",
    ".data": "application/octet-stream",
    ".json": "application/json",
    ".png": "image/png",
    ".ico": "image/x-icon",
}


class _SessionMessage(enum.StrEnum):
  """Text messages browsers send on /state."""

  REQUEST_CONTROL = "request_control"
  LEAVE_QUEUE = "leave_queue"
  FORCE_CONTROL = "force_control"
  HEARTBEAT = "heartbeat"


def _roster_line(
    viewers: int,
    is_controller: bool,
    queue_pos: int,
    queue_len: int,
    max_spectators: int,
) -> str:
  """Builds a roster line, the membership broadcast every browser receives.

  The roster is sent as a text frame on /state whenever the session changes (a
  viewer joins or leaves, queues for control, or control moves). It tells each
  browser how many viewers are connected, which role the server has currently
  assigned it, and where it stands in the control queue.

  Args:
    viewers: The number of viewers connected to the server.
    is_controller: Whether the browser is the controller.
    queue_pos: The browser's position in the control queue (0 if not queued).
    queue_len: The number of browsers in the queue.
    max_spectators: The maximum number of spectators allowed.

  Returns:
    A string representing the roster line.
  """
  role = "controller" if is_controller else "spectator"
  return (
      f"viewers={viewers};role={role};queue_pos={queue_pos};"
      f"queue_len={queue_len};max_spectators={max_spectators}"
  )


class _WebServerFormatter(logging.Formatter):

  def format(self, record: logging.LogRecord) -> str:
    level_char = record.levelname[0]
    dt = datetime.datetime.fromtimestamp(record.created)
    date_str = dt.strftime("%m%d")
    time_str = dt.strftime("%H:%M:%S.%f")
    filename = os.path.basename(record.pathname)
    line = record.lineno
    msg = record.getMessage()
    return f"{level_char}{date_str} {time_str} {filename}:{line}] {msg}"


logger = logging.getLogger("WebServer")


class _HandshakeNoiseFilter(logging.Filter):
  """Removes handshake-failure logs from connections that simply went away.

  Handshake failure are produced by two routine cases:

  1. browsers speculatively open spare connections and close them unused
  2. a page reload on model change tears connections down mid-handshake.

  The websockets library logs each as an opening-handshake failure as at ERROR
  level with a full traceback. Raising the log level would hide real handshake
  errors too, hence a filter matching exactly these signatures.
  """

  def filter(self, record: logging.LogRecord) -> bool:
    if record.getMessage() != "opening handshake failed":
      return True
    exc = record.exc_info[1] if record.exc_info else None
    if exc is None:
      return True
    if isinstance(exc, (ConnectionClosed, InvalidHandshake, EOFError, OSError)):
      return False  # The connection died or sent non-WS HTTP; nothing wrong.
    return "did not receive a valid HTTP request" not in str(exc)


def _configure_logging() -> None:
  """Configures this module's logger and reins in the websockets library's.

  Called once at import time; a no-op if already configured (re-import).
  """
  if logger.handlers:
    return
  # Info is a curated, always-on channel (browser connect/disconnect);
  # connection plumbing is logged at DEBUG.
  logger.setLevel(logging.INFO)
  handler = logging.StreamHandler(sys.stdout)
  handler.setFormatter(_WebServerFormatter())
  logger.addHandler(handler)
  logger.propagate = False
  # The websockets library logs its own lifecycle chatter at INFO ("server
  # listening", "connection rejected (200 OK)" for every HTTP request); keep
  # it to warnings and errors.
  logging.getLogger("websockets").setLevel(logging.WARNING)
  # Filters do not inherit down the logger tree, so it must be attached to
  # "websockets.server" itself.
  logging.getLogger("websockets.server").addFilter(_HandshakeNoiseFilter())


_configure_logging()


def bind_public_socket(host: str, port: int = 0) -> socket.socket:
  """Binds (but does not listen on) the public HTTP listening socket.

  The socket is bound in the viewer thread up front so that bind conflicts
  surface here, synchronously, rather than inside the server thread.

  Args:
    host: Interface to bind. "::" (the default) binds a dual-stack socket that
      accepts both IPv6 and IPv4 connections, falling back to IPv4-only when the
      machine has no IPv6 support.
    port: A specific port, or 0 to take the first free port starting at
      _DEFAULT_HTTP_PORT (so several viewers can run side by side).

  Returns:
    The bound socket.

  Raises:
    RuntimeError: If the requested port (or every scanned port) is taken.
  """
  if host == "::":
    try:
      socket.socket(socket.AF_INET6, socket.SOCK_STREAM).close()
    except OSError:
      host = "0.0.0.0"
  family = socket.AF_INET6 if ":" in host else socket.AF_INET

  candidates = (
      [port]
      if port
      else range(_DEFAULT_HTTP_PORT, _DEFAULT_HTTP_PORT + _PORT_SCAN_COUNT)
  )
  for candidate in candidates:
    sock = socket.socket(family, socket.SOCK_STREAM)
    sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    if family == socket.AF_INET6:
      # Dual-stack: also accept IPv4 connections (as IPv4-mapped addresses).
      sock.setsockopt(socket.IPPROTO_IPV6, socket.IPV6_V6ONLY, 0)
    try:
      sock.bind((host, candidate))
      return sock
    except OSError:
      sock.close()
  if port:
    raise RuntimeError(
        f"Port {port} is already in use, is another web viewer running? "
        "Pass a different --port, or 0 to pick one automatically."
    )
  raise RuntimeError(
      f"Ports {_DEFAULT_HTTP_PORT}-{_DEFAULT_HTTP_PORT + _PORT_SCAN_COUNT - 1} "
      "are all in use, are that many web viewers running?"
  )


def bind_loopback_socket(port: int = 0) -> socket.socket:
  """Binds the loopback socket the headless NetImgui client connects to.

  Defaults to an OS-assigned ephemeral port: both endpoints live in this process
  tree, so no fixed number is needed and collisions between viewer instances are
  impossible.

  Args:
    port: A specific port, or 0 to take the first free port.

  Returns:
    The bound, listening socket.
  """
  sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
  sock.bind(("127.0.0.1", port))
  sock.listen()
  return sock


def _session_id(ws: ServerConnection) -> str:
  """The page's session id (?sid=...), tying its /ui and /state together."""
  path = ws.request.path if ws.request else ""
  query = path.split("?", 1)[1] if "?" in path else ""
  for part in query.split("&"):
    if part.startswith("sid="):
      return part[4:]
  return f"anon-{id(ws)}"


def _close_reason(text: str) -> str:
  """Fits text into a close frame's reason, cutting at a character boundary."""
  return text.encode()[:_WS_CLOSE_REASON_MAX_BYTES].decode(errors="ignore")


def _find_static_files_dir() -> str | None:
  """Locate the web viewer static files.

  The `dist` directory next to this file is populated by the Emscripten build
  of the `web_client` target. This is the default for packaged installations.

  Development option: Set MUJOCO_WEB_VIEWER_DIST to serve from a different
  directory (e.g. a local Emscripten build tree). This allows rapid iteration
  on web_client changes without rebuilding and reinstalling the package.
  """
  env_dir = os.environ.get("MUJOCO_WEB_VIEWER_DIST")
  if env_dir:
    if os.path.isdir(env_dir):
      return env_dir
    logger.warning(
        "[Http] MUJOCO_WEB_VIEWER_DIST is not a directory: %s", env_dir
    )

  dist_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "dist")
  if os.path.isdir(dist_dir):
    return dist_dir

  return None


def _run_cancellable(main_loop_func: Callable[[], Awaitable[None]]) -> None:
  """Runs an asyncio loop, logging unexpected errors."""
  try:
    asyncio.run(main_loop_func())
  except asyncio.CancelledError:
    pass
  except Exception as e:  # pylint: disable=broad-exception-caught
    func_name = getattr(main_loop_func, "__name__", "main_loop_func")
    logger.error("[%s] Unexpected error: %s", func_name, e, exc_info=True)


def _run_server(
    server: "WebServer",
    stop_event: threading.Event,
) -> None:
  """The server thread: HTTP + /ui + /state on one port, one event loop."""

  http_sock = server.http_sock
  tcp_sock = server.tcp_sock
  static_dir = server.static_files_dir
  static_root = os.path.realpath(static_dir) if static_dir else None

  def _http_headers(
      content_type: str, content_length: int, cacheable: bool
  ) -> Headers:
    headers = Headers()
    headers["Content-Type"] = content_type
    headers["Content-Length"] = str(content_length)
    # The WASM is a pthread build: SharedArrayBuffer requires cross-origin
    # isolation on every response.
    headers["Cross-Origin-Opener-Policy"] = "same-origin"
    headers["Cross-Origin-Embedder-Policy"] = "require-corp"
    if not cacheable:
      headers["Access-Control-Allow-Origin"] = "*"
      headers["Cache-Control"] = "no-store"
    return headers

  def _serve_http(path: str) -> Response:
    """Builds the HTTP response for a non-WebSocket GET request."""
    if path == "/model":
      mjb_data = server.mjb_data  # Snapshot to prevent race with update_model()
      if not mjb_data:
        return Response(404, "Not Found", Headers(), b"no model\n")
      # The model changes on hot-swap; never serve a cached copy.
      headers = _http_headers(
          "application/octet-stream", len(mjb_data), cacheable=False
      )
      return Response(200, "OK", headers, mjb_data)

    if static_root is None:
      return Response(503, "Service Unavailable", Headers(), b"no dist dir\n")

    rel = path.lstrip("/") or "index.html"
    norm_static_root = os.path.normpath(static_root)
    full = os.path.normpath(os.path.join(norm_static_root, rel))
    if full != norm_static_root and not full.startswith(
        norm_static_root + os.sep
    ):
      return Response(403, "Forbidden", Headers(), b"forbidden\n")
    if not os.path.isfile(full):
      wasm_full = os.path.normpath(os.path.join(norm_static_root, "wasm", rel))
      studio_root = os.path.normpath(
          os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
      )
      studio_assets_full = os.path.normpath(os.path.join(studio_root, rel))
      if os.path.isfile(wasm_full):
        full = wasm_full
      elif (
          rel.startswith("assets/")
          and studio_assets_full.startswith(
              os.path.join(studio_root, "assets") + os.sep
          )
          and os.path.isfile(studio_assets_full)
      ):
        full = studio_assets_full
      else:
        return Response(404, "Not Found", Headers(), b"not found\n")

    with open(full, "rb") as f:
      body = f.read()
    content_type = _CONTENT_TYPES.get(
        os.path.splitext(full)[1], "application/octet-stream"
    )
    cacheable = not rel.endswith("index.html")
    return Response(
        200,
        "OK",
        _http_headers(content_type, len(body), cacheable=cacheable),
        body,
    )

  async def main_loop() -> None:
    # The NetImgui client connection (from HeadlessUi, loopback TCP). The client
    # retries every second, so after a teardown a fresh connection is fast.
    tcp_reader = None
    tcp_writer = None
    tcp_connected = asyncio.Event()

    # The browser controlling the UI (single slot) and all browsers receiving
    # the state broadcast (controller + spectators), keyed by session id.
    active_ui_ws = None
    controller_sid: str | None = None
    state_clients: dict[str, ServerConnection] = {}

    # Wakes every /state stream when the viewer thread publishes a payload (see
    # WebServer.update_state), so a payload goes out the moment it exists. A
    # 60Hz poll instead would free-run against the viewer's 60Hz frame clock
    # and periodically cost a whole tick. One event per stream: each stream
    # clears its own, so a shared one would lose wakeups for the others.
    state_waiters: set[asyncio.Event] = set()

    def wake_state_streams() -> None:
      for event in state_waiters:
        event.set()

    server._wake_state_streams = functools.partial(
        asyncio.get_running_loop().call_soon_threadsafe, wake_state_streams
    )

    # Control handoff: spectators queue for the controller slot; when it frees,
    # the head of the queue is granted a short exclusive claim window.
    control_queue: list[str] = []
    pending_grant_sid: str | None = None
    broadcast_roster: Any = None
    grant_next: Any = None

    # Strong references to fire-and-forget tasks (the grant-expiry watchdog):
    # asyncio holds only a weak reference, so without this a task can be
    # garbage-collected mid-sleep and never run.
    background_tasks: set[asyncio.Task[Any]] = set()
    max_spectators = _MAX_SPECTATORS_DEFAULT

    # Liveness by absence of traffic: a hidden tab's rendering loop stops,
    # so it cannot report anything — silence is the signal.
    last_heartbeat: dict[str, float] = {}
    last_controller_input = [0.0]
    loop_time = asyncio.get_event_loop().time

    async def broadcast_roster() -> None:
      """Sends every browser the counts, its role and its queue position."""
      count = len(state_clients)
      for sid, client in list(state_clients.items()):
        try:
          pos = control_queue.index(sid) + 1
        except ValueError:
          pos = 0
        try:
          await client.send(
              _roster_line(
                  viewers=count,
                  is_controller=sid == controller_sid,
                  queue_pos=pos,
                  queue_len=len(control_queue),
                  max_spectators=max_spectators,
              )
          )
        except (ConnectionClosed, ConnectionError):
          pass

    async def grant_next() -> None:
      """Offers the free controller slot to the pending or next queued page."""
      nonlocal pending_grant_sid
      if active_ui_ws is not None:
        return

      if pending_grant_sid is None:
        while control_queue:
          candidate = control_queue.pop(0)
          if candidate in state_clients:
            pending_grant_sid = candidate
            break
      if pending_grant_sid is None:
        return

      client = state_clients.get(pending_grant_sid)
      if client is None:
        pending_grant_sid = None
        await grant_next()
        return

      try:
        await client.send(_GRANT_MESSAGE)
      except (ConnectionClosed, ConnectionError):
        pending_grant_sid = None
        await grant_next()
        return
      logger.info("[Session] Control granted to %s", pending_grant_sid)
      await broadcast_roster()

      async def expire(granted: str) -> None:
        nonlocal pending_grant_sid
        await asyncio.sleep(_GRANT_EXPIRY_SEC)
        if pending_grant_sid == granted and active_ui_ws is None:
          pending_grant_sid = None
          await grant_next()

      task = asyncio.create_task(expire(pending_grant_sid))
      background_tasks.add(task)
      task.add_done_callback(background_tasks.discard)

    async def handle_session_message(sid: str, text: str) -> None:
      """A control or heartbeat message from one browser (/state text)."""
      nonlocal pending_grant_sid, max_spectators
      last_heartbeat[sid] = loop_time()
      if text == _SessionMessage.REQUEST_CONTROL:
        if sid != controller_sid and sid not in control_queue:
          control_queue.append(sid)
          await grant_next()
          await broadcast_roster()
      elif text == _SessionMessage.LEAVE_QUEUE:
        if sid in control_queue:
          control_queue.remove(sid)
          await broadcast_roster()
      elif text == _SessionMessage.FORCE_CONTROL:
        if sid != controller_sid:
          logger.info("[Session] %s forces control", sid)
          if sid in control_queue:
            control_queue.remove(sid)
          pending_grant_sid = sid
          if active_ui_ws is not None:
            # The bridge teardown frees the slot and grant_next honors the
            # pending claim; the ousted page settles into spectating.
            await active_ui_ws.close(
                _WS_CLOSE_CONTROLLER_TAKEN, "control taken"
            )
          else:
            await grant_next()
      elif text == _SessionMessage.HEARTBEAT:
        pass  # last_heartbeat is updated for every message.
      elif text.startswith(_MAX_SPECTATORS_PREFIX):
        if sid != controller_sid:
          return  # Only the controller sets the limit.
        try:
          value = int(text[len(_MAX_SPECTATORS_PREFIX) :])
        except ValueError:
          return
        max_spectators = max(0, min(_MAX_SPECTATOR_HARD_CAP, value))
        logger.info("[Session] Spectator limit set to %s", max_spectators)
        # Enforce the new limit immediately, kicking the newest spectators
        # first. A kicked page shows the session-full notice and retries,
        # so it walks back in when a slot frees up.
        excess = len(state_clients) - (max_spectators + 1)
        for kick_sid in reversed(list(state_clients)):
          if excess <= 0:
            break
          if kick_sid == controller_sid:
            continue
          client = state_clients.get(kick_sid)
          if client is not None:
            logger.info("[Session] Kicking spectator over the new limit")
            try:
              await client.close(_WS_CLOSE_SESSION_FULL, "session full")
            except (ConnectionClosed, ConnectionError):
              pass
          excess -= 1
        await broadcast_roster()

    async def enforce_activity() -> None:
      """Releases a silent controller when someone waits; kicks silent tabs."""
      while True:
        await asyncio.sleep(5.0)
        now = loop_time()
        if (
            active_ui_ws is not None
            and last_controller_input[0] > 0
            and now - last_controller_input[0] > _CONTROLLER_SILENT_KICK_SEC
            and (control_queue or pending_grant_sid)
        ):
          logger.info("[Session] Releasing control from an inactive controller")
          await active_ui_ws.close(_WS_CLOSE_CONTROLLER_TAKEN, "inactive")
        for sid, client in list(state_clients.items()):
          if sid == controller_sid:
            continue
          seen = last_heartbeat.get(sid, now)
          if now - seen > _SPECTATOR_SILENT_KICK_SEC:
            logger.info("[Session] Kicking inactive spectator %s", sid)
            try:
              await client.close(_WS_CLOSE_INACTIVE, "inactive")
            except (ConnectionClosed, ConnectionError):
              pass

    async def handle_tcp_client(
        reader: asyncio.StreamReader, writer: asyncio.StreamWriter
    ) -> None:
      nonlocal tcp_reader, tcp_writer
      if tcp_writer is not None:
        logger.debug("[UiBridge] Replacing previous NetImgui TCP connection")
        tcp_writer.close()
      tcp_reader = reader
      tcp_writer = writer
      # Disable Nagle's algorithm so small packets (CmdInput ~80 bytes) are
      # sent immediately instead of being buffered for up to 200ms.
      sock = writer.get_extra_info("socket")
      if sock:
        sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
      tcp_connected.set()
      logger.debug("[UiBridge] NetImgui TCP connection accepted")

    # --- /ui: bridge the browser to the NetImgui client ----------------------

    async def ui_handler(ws: ServerConnection) -> None:
      nonlocal active_ui_ws, controller_sid, pending_grant_sid
      nonlocal tcp_reader, tcp_writer
      if active_ui_ws is not None:
        # One controller at a time. The rejected browser spectates and may retry
        # via its "Take control" button once the controller leaves.
        logger.debug("[UiBridge] Controller slot taken; rejecting new browser")
        await ws.close(_WS_CLOSE_CONTROLLER_TAKEN, "controller slot taken")
        return
      sid = _session_id(ws)
      if pending_grant_sid is not None and sid != pending_grant_sid:
        # The slot is reserved for a granted page (queue fairness).
        await ws.close(_WS_CLOSE_CONTROLLER_TAKEN, "controller slot reserved")
        return
      if sid == pending_grant_sid:
        pending_grant_sid = None
      if sid in control_queue:
        control_queue.remove(sid)
      active_ui_ws = ws
      controller_sid = sid
      last_controller_input[0] = loop_time()
      await broadcast_roster()

      # Each browser needs a fresh NetImgui session (full handshake, textures
      # and draw state). Drop any connection used by a previous bridge and
      # wait for the client's automatic reconnect.
      if tcp_writer is not None:
        tcp_writer.close()
        tcp_reader = None
        tcp_writer = None
        tcp_connected.clear()
      logger.debug("[UiBridge] Waiting for NetImgui TCP connection...")

      async def wait_for_tcp() -> None:
        # Loop: the connection can be replaced (and the event cleared)
        # between the event firing and this task waking up.
        while tcp_writer is None:
          await tcp_connected.wait()

      # Wait for the client, but not forever and not blind to the browser: if
      # the client never appears (bounded by _UI_TCP_WAIT_SEC) or the browser
      # leaves first, release the controller slot instead of holding it
      # against every future browser.
      tcp_task = asyncio.create_task(wait_for_tcp())
      close_task = asyncio.create_task(ws.wait_closed())
      done, still_pending = await asyncio.wait(
          [tcp_task, close_task],
          timeout=_UI_TCP_WAIT_SEC,
          return_when=asyncio.FIRST_COMPLETED,
      )
      for task in still_pending:
        task.cancel()
      if tcp_task not in done or tcp_task.cancelled():
        logger.debug("[UiBridge] No NetImgui connection; releasing slot")
        if active_ui_ws is ws:
          active_ui_ws = None
          controller_sid = None
          await grant_next()
          await broadcast_roster()
        await ws.close(_WS_CLOSE_CONTROLLER_TAKEN, "controller unavailable")
        return
      if tcp_reader is None or tcp_writer is None:
        logger.debug("[UiBridge] NetImgui reader/writer None; releasing slot")
        if active_ui_ws is ws:
          active_ui_ws = None
          controller_sid = None
          await grant_next()
          await broadcast_roster()
        await ws.close(_WS_CLOSE_CONTROLLER_TAKEN, "controller unavailable")
        return
      my_reader = cast(asyncio.StreamReader, tcp_reader)
      my_writer = cast(asyncio.StreamWriter, tcp_writer)

      try:
        # Handshake: browser CmdVersion -> client; client CmdVersion -> browser.
        browser_version = await ws.recv()
        my_writer.write(browser_version)  # pyrefly: ignore[bad-argument-type]
        await my_writer.drain()
        server_version = await my_reader.readexactly(_NETIMGUI_CMD_VERSION_SIZE)
        await ws.send(server_version)
        logger.debug("[UiBridge] Handshake complete. Bridging.")

        async def ws_to_tcp() -> None:
          async for message in ws:
            if isinstance(message, bytes):
              last_controller_input[0] = loop_time()
              my_writer.write(message)
              await my_writer.drain()

        async def tcp_to_ws() -> None:
          while True:
            data = await my_reader.read(65536)
            if not data:
              break
            await ws.send(data)

        done, pending = await asyncio.wait(
            [
                asyncio.create_task(ws_to_tcp()),
                asyncio.create_task(tcp_to_ws()),
            ],
            return_when=asyncio.FIRST_COMPLETED,
        )
        for task in pending:
          task.cancel()
        for task in done:
          task.exception()  # Retrieve to avoid "exception never retrieved".
      except (ConnectionClosed, ConnectionError, asyncio.IncompleteReadError):
        pass
      finally:
        logger.debug("[UiBridge] Bridge closed.")
        # Tear down this bridge's TCP connection so the NetImgui client
        # reconnects and re-sends everything to the next browser.
        if my_writer is not None:
          my_writer.close()
        if tcp_writer is my_writer:
          tcp_reader = None
          tcp_writer = None
          tcp_connected.clear()
        if active_ui_ws is ws:
          active_ui_ws = None
          controller_sid = None
          await grant_next()
          await broadcast_roster()

    # --- /state: latest-wins payload broadcast --------------------------------

    async def state_handler(ws: ServerConnection) -> None:
      sid = _session_id(ws)
      # A page reconnecting after a network blip keeps its sid; its stale entry
      # (dead socket, not yet cleaned up) must not count against the limit,
      # since installing the new socket merely replaces it. Only a genuinely new
      # sid consumes a slot.
      if sid not in state_clients and len(state_clients) > max_spectators:
        logger.info("[StateWS] Session full; rejecting browser")
        await ws.close(_WS_CLOSE_SESSION_FULL, "session full")
        return
      state_clients[sid] = ws
      last_heartbeat[sid] = loop_time()
      logger.info("[StateWS] Browser connected (%d total)", len(state_clients))
      await broadcast_roster()

      def read_state() -> tuple[int, bytes] | None:
        """Reads the latest state payload from the server instance.

        In the thread model, the viewer thread writes server._state_payload and
        bumps server._state_generation (via WebServer.update_state()) under
        server._state_lock; this side just grabs a snapshot.  Returns
        (generation, payload) or None.
        """
        with server._state_lock:
          gen = server._state_generation
          data = server._state_payload
        if data is None:
          return None
        return gen, data

      # See _STATE_ACK_MESSAGE: one payload in flight at a time, so the latest
      # wins read below always ships the freshest state a slow link can carry
      # (the camera rides this stream, so socket buffering shows up directly as
      # input latency).
      payload_acked = asyncio.Event()
      payload_acked.set()
      # Set initially so a browser that connects between payloads gets the
      # current one right away.
      state_ready = asyncio.Event()
      state_ready.set()
      state_waiters.add(state_ready)

      async def send_payloads() -> None:
        last_gen = 0
        while True:
          await state_ready.wait()
          try:
            await asyncio.wait_for(
                payload_acked.wait(), timeout=_STATE_ACK_TIMEOUT_SEC
            )
          except asyncio.TimeoutError:
            pass  # Lost ack; send anyway rather than stalling forever.
          # Cleared before the read: an update racing with it sets the event
          # again and is picked up on the next pass instead of being lost.
          state_ready.clear()
          result = read_state()
          if result is None or result[0] == last_gen:
            continue
          last_gen, data = result
          payload_acked.clear()
          await ws.send(data)

      async def recv_messages() -> None:
        async for message in ws:
          if isinstance(message, str):
            if message == _STATE_ACK_MESSAGE:
              # Deliberately does NOT count as liveness: acks keep flowing from
              # hidden tabs (WebSocket events are not requestAnimationFrame
              # gated), and the inactivity policy must still see those tabs as
              # silent.
              payload_acked.set()
              continue
            await handle_session_message(sid, message)

      try:
        done, pending = await asyncio.wait(
            [
                asyncio.create_task(send_payloads()),
                asyncio.create_task(recv_messages()),
            ],
            return_when=asyncio.FIRST_COMPLETED,
        )
        for task in pending:
          task.cancel()
        for task in done:
          task.exception()  # Retrieve to avoid "exception never retrieved".
      except (ConnectionClosed, ConnectionError):
        pass
      finally:
        state_waiters.discard(state_ready)
        # Only tear down if this socket is still the one registered for sid: a
        # reconnect may have already replaced it, and that newer connection now
        # owns the queue and heartbeat entries.
        if state_clients.get(sid) is ws:
          del state_clients[sid]
          if sid in control_queue:
            control_queue.remove(sid)
          last_heartbeat.pop(sid, None)
          logger.info(
              "[StateWS] Browser disconnected (%d total)", len(state_clients)
          )
          await broadcast_roster()

    # --- Dispatch ----------------------------------------------------------

    def process_request(
        connection: ServerConnection, request: Request
    ) -> Response | None:
      del connection

      # Split path and query string.
      if "?" in request.path:
        path, query_string = request.path.split("?", 1)
      else:
        path, query_string = request.path, ""
      if path in ("/ui", "/state", "/drop"):
        # Plain HTTP requests to a WebSocket path (e.g. browser speculative probe
        # or proxy without Upgrade header) get a clean 426 Upgrade Required
        # instead of triggering websockets' opening-handshake exception log.
        connection_opts = [
            opt.strip().lower()
            for val in request.headers.get_all("Connection")
            for opt in val.split(",")
        ]
        if "upgrade" not in connection_opts:
          headers = Headers()
          headers["Upgrade"] = "websocket"
          headers["Connection"] = "Upgrade"
          return Response(
              426,
              "Upgrade Required",
              headers,
              b"WebSocket upgrade required\n",
          )
        return None  # Proceed with the WebSocket handshake.

      # Chunked model endpoint: the client fetches /model in parallel chunks
      #
      #   GET /model?total_bytes                 -> {"total_bytes": <n>}
      #   GET /model?offset_bytes=X&size_bytes=Y -> bytes [X, X+Y)
      #
      # Full model endpoint: the client fetches /model in a single request
      #
      #   GET /model  -> full model bytes
      mjb_data = server.mjb_data  # Snapshot to prevent race with update_model()
      if path == "/model" and mjb_data and query_string:
        params = urllib.parse.parse_qs(query_string)
        if "total_bytes" in params or query_string == "total_bytes":
          body = json.dumps({"total_bytes": len(mjb_data)}).encode()
          headers = _http_headers(
              "application/json", len(body), cacheable=False
          )
          return Response(200, "OK", headers, body)
        try:
          offset = int(params.get("offset_bytes", [0])[0])
          size = int(params.get("size_bytes", [0])[0])
        except (ValueError, TypeError):
          return Response(400, "Bad Request", Headers(), b"bad params\n")
        if size <= 0 or offset < 0 or offset >= len(mjb_data):
          return Response(400, "Bad Request", Headers(), b"bad range\n")
        chunk = mjb_data[offset : min(offset + size, len(mjb_data))]
        headers = _http_headers(
            "application/octet-stream", len(chunk), cacheable=False
        )
        return Response(200, "OK", headers, chunk)

      return _serve_http(path)

    def process_response(
        connection: ServerConnection, request: Request, response: Response
    ) -> Response | None:
      """Adds cross-origin headers to the WebSocket upgrade (101) response.

      The page serves Cross-Origin-Opener-Policy: same-origin and
      Cross-Origin-Embedder-Policy: require-corp so the WASM pthread build
      can use SharedArrayBuffer. When accessed through an HTTPS reverse proxy,
      the browser's cross-origin isolation policy blocks WebSocket connections
      unless the upgrade response carries matching headers. Without them, the
      handshake fails with close code CloseCode.ABNORMAL_CLOSURE.

      Args:
        connection: The WebSocket server connection (unused).
        request: The initial HTTP upgrade request (unused).
        response: The HTTP upgrade response to attach CORS headers to.

      Returns:
        None to allow the WebSocket upgrade handshake to proceed.
      """
      del connection
      del request
      response.headers["Cross-Origin-Opener-Policy"] = "same-origin"
      response.headers["Cross-Origin-Embedder-Policy"] = "require-corp"
      response.headers["Access-Control-Allow-Origin"] = "*"
      response.headers["Cross-Origin-Resource-Policy"] = "cross-origin"
      return None

    # --- /drop: receive a file dropped onto the browser page ---------------

    async def drop_handler(ws: ServerConnection) -> None:
      """Receives dropped files and forwards them to the viewer process.

      The browser opens this socket on demand and sends one binary frame per
      file ([u32 path length][relative path utf-8][file bytes], see
      index.html), then an empty frame as an end marker; this server closes
      once it has everything. The files cross to the viewer process via
      drop_queue as a dict of relative path -> bytes; WebViewer.get_drop_file()
      writes them to a temporary directory for the regular drop-loading flow.

      Loading a model changes the session for every connected page, so only
      the controller may do it; drops from other pages are rejected.

      Args:
        ws: The WebSocket connection to receive from.
      """
      if controller_sid is None or _session_id(ws) != controller_sid:
        logger.info("[Drop] Ignored a model drop from a non-controller page")
        await ws.close(_WS_CLOSE_NOT_CONTROLLER, "not the controller")
        return

      files: dict[str, bytes] = {}
      complete = False
      try:
        async for message in ws:
          if not isinstance(message, bytes):
            continue
          if not message:
            complete = True  # End-of-drop marker.
            break
          if len(message) < 4:
            continue
          (name_len,) = struct.unpack_from("<I", message)
          if 4 + name_len > len(message):
            continue
          name = message[4 : 4 + name_len].decode("utf-8", "replace")
          files[name] = message[4 + name_len :]
      except (ConnectionClosed, ConnectionError):
        pass

      # Only forward a drop that arrived in full. A connection that dies before
      # the end marker (e.g. a file frame exceeding the websocket size cap
      # closes it with CloseCode.MESSAGE_TOO_BIG) leaves a partial file set that
      # would load as a broken model, so discard it instead.
      if not complete:
        if files:
          logger.info(
              "[Drop] Incomplete drop (%d file(s) before the"
              " connection closed); discarding.",
              len(files),
          )
      elif files and server.drop_queue is not None:
        total = sum(len(data) for data in files.values())
        logger.info("[Drop] Received %d file(s), %d bytes", len(files), total)
        server.drop_queue.put(files)

      await ws.close()

    async def ws_handler(ws: ServerConnection) -> None:
      req = ws.request
      if req is None:
        await ws.close(CloseCode.POLICY_VIOLATION, "missing request")
        return
      path = req.path.split("?")[0]
      if path == "/ui":
        await ui_handler(ws)
      elif path == "/state":
        await state_handler(ws)
      elif path == "/drop":
        await drop_handler(ws)
      else:
        await ws.close(CloseCode.POLICY_VIOLATION, "unknown endpoint")

    # Block until the stop event is set by WebServer.stop(). The server thread
    # is a daemon, so it also dies if the process exits for any reason.
    async def watch_stop() -> None:
      loop = asyncio.get_running_loop()
      stopped = asyncio.Event()

      def poll() -> None:
        stop_event.wait()
        try:
          loop.call_soon_threadsafe(stopped.set)
        except RuntimeError:
          pass  # Loop already closed.

      threading.Thread(target=poll, daemon=True).start()
      await stopped.wait()
      logger.debug("[Http] Stop event received; shutting down.")

    tcp_server = await asyncio.start_server(handle_tcp_client, sock=tcp_sock)

    # NetImgui does its own delta compression and the state payload is small;
    # permessage-deflate would only add latency at 60Hz.
    ws_server = await serve(
        ws_handler,
        sock=http_sock,
        process_request=process_request,
        process_response=process_response,
        compression=None,
        ping_interval=None,
        ping_timeout=None,
        open_timeout=None,
        close_timeout=None,
        # Big enough for model files uploaded via /drop.
        max_size=2**26,
    )
    http_host, http_port = http_sock.getsockname()[:2]
    tcp_port = tcp_sock.getsockname()[1]
    logger.debug(
        "[Http] Serving on http://%s:%d "
        "(/, /model, /ui, /state; NetImgui TCP on 127.0.0.1:%d)",
        http_host,
        http_port,
        tcp_port,
    )

    enforcer = asyncio.create_task(enforce_activity())
    try:
      await watch_stop()
    finally:
      server._wake_state_streams = None
      enforcer.cancel()
      if server.stop_reason:
        # Tell the browsers why the viewer stopped, so the page can show it;
        # ws_server.close() below only says "going away". Bounded: a browser
        # that never answers the close handshake must not hold up teardown.
        reason = _close_reason(server.stop_reason)
        try:
          await asyncio.wait_for(
              asyncio.gather(
                  *(
                      ws.close(_WS_CLOSE_VIEWER_ERROR, reason)
                      for ws in list(state_clients.values())
                  ),
                  return_exceptions=True,
              ),
              timeout=1.0,
          )
        except asyncio.TimeoutError:
          pass
      # Close listeners explicitly to prevent hanging on open client sockets.
      ws_server.close()
      tcp_server.close()

  _run_cancellable(main_loop)


class WebServer:
  """The Web Viewer's single-port server.

  Serves the browser page, WASM, and model over HTTP; bridges the NetImgui UI
  stream at /ui; and broadcasts the state payload at /state on a single public
  port from a background daemon thread.  State payloads are handed over through
  a lock-protected reference with latest-wins semantics: update_state() may be
  called at any rate from the viewer thread; browsers only ever see the newest
  payload.  The server thread is started once and persists.  When the model
  changes, update_model() swaps the bytes served at /model, without any restart.

  Run the simulation with the web viewer, then visit the printed URL.
  """

  def __init__(
      self,
      http_sock: socket.socket,
      tcp_sock: socket.socket,
      static_files_dir: str | None = None,
      mjb_data: bytes | None = None,
      drop_queue: queue.Queue[Any] | None = None,
  ) -> None:
    """Initializes the server around pre-bound listening sockets.

    The sockets are bound by the viewer (see bind_public_socket /
    bind_loopback_socket) and shared with the server thread, so bind failures
    can't occur here.

    Args:
      http_sock: The listening socket for HTTP and WebSocket traffic.
      tcp_sock: The listening socket for NetImgui traffic.
      static_files_dir: The directory containing the static files to serve.
      mjb_data: The model data to serve from /model.
      drop_queue: The queue to put dropped files onto.
    """
    self.http_sock = http_sock
    self.tcp_sock = tcp_sock
    self.static_files_dir = static_files_dir or _find_static_files_dir()
    self.mjb_data = mjb_data

    # Dropped-file bytes travel from the server thread back to the viewer thread
    # through this queue.
    self.drop_queue = drop_queue

    # State payload: the viewer thread writes, the server thread reads.
    self._state_lock = threading.Lock()
    self._state_payload: bytes | None = None
    self._state_generation: int = 0
    self._wake_state_streams: Callable[[], Any] | None = None

    self._stop_event = threading.Event()
    self._thread: threading.Thread | None = None
    # Why the viewer stopped, when not normally (see stop()); read by the
    # server thread as it shuts down.
    self.stop_reason = ""

  def update_state(self, payload: bytes) -> None:
    """Publishes the latest state payload. Called from the viewer thread."""
    with self._state_lock:
      self._state_payload = payload
      self._state_generation += 1
    # Hop to the server thread's event loop to wake the /state streams; unset
    # while the server thread is not running.
    wake = self._wake_state_streams
    if wake is not None:
      try:
        wake()
      except RuntimeError:
        pass  # The server's event loop has just closed.

  def update_model(self, mjb_data: bytes) -> None:
    """Swaps the model bytes served at /model. No server restart needed."""
    self.mjb_data = mjb_data

  def start(self) -> None:
    """Starts the server on a background daemon thread."""
    if not self.static_files_dir:
      logger.warning("[Http] WARNING: Could not find web viewer static files.")
      return

    logger.debug("[Http] Serving web viewer from: %s", self.static_files_dir)

    self._stop_event.clear()
    self._thread = threading.Thread(
        target=_run_server,
        args=(self, self._stop_event),
        daemon=True,
    )
    self._thread.start()

  def stop(self, reason: str = "") -> None:
    """Stops the server thread.

    Args:
      reason: Why the viewer stopped, when this is not a normal exit: a one-line
        summary of the exception that ended it. Sent to the connected browsers,
        which show it on the page.
    """
    self.stop_reason = reason
    self._stop_event.set()
    if self._thread is not None:
      self._thread.join(timeout=5.0)
      self._thread = None
