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

// The Python side of the web viewer's UI stream.
//
// HeadlessUiSession turns the frames of a headless Dear ImGui context into
// Studio UI wire messages for the browser (textures, clipboards, and
// delta-compressed draw frames) and converts browser Input messages into Dear
// ImGui input events.
//
// Threading: the UI thread calls ApplyInput(), CaptureFrame(), and
// QueueClipboard(); the web server thread calls StartSession(), EndSession(),
// IsActive(), HasInput(), PushInput(), TakeMessages(), and LastFrameIndex().
// All Dear ImGui calls and frame encoding happen on the UI thread.

#ifndef MUJOCO_PYTHON_STUDIO_WEB_HEADLESS_UI_SESSION_H_
#define MUJOCO_PYTHON_STUDIO_WEB_HEADLESS_UI_SESSION_H_

#include <cstddef>
#include <cstdint>
#include <mutex>
#include <string>
#include <string_view>
#include <unordered_set>
#include <vector>

#include <imgui.h>
#include "ui_codec.h"

namespace mujoco::studio {

class HeadlessUiSession {
 public:
  HeadlessUiSession() = default;
  ~HeadlessUiSession();

  HeadlessUiSession(const HeadlessUiSession&) = delete;
  HeadlessUiSession& operator=(const HeadlessUiSession&) = delete;

  // ---- Server thread --------------------------------------------------------

  // Starts a session for a newly connected browser from its Hello handshake
  // message. Populates `reply` on success or `error` on failure.
  bool StartSession(netui::BytesView handshake, netui::Bytes* reply,
                    std::string* error);

  // Ends the session. No frames are captured until the next StartSession().
  void EndSession();

  // True between a successful StartSession() and EndSession().
  bool IsActive() const;

  // True once the active session's browser has sent input.
  bool HasInput() const;

  // Queues one Input message from the browser for the next ApplyInput().
  // Returns false if the message is invalid or no session is active.
  bool PushInput(netui::BytesView message);

  // Returns pending texture, clipboard, and draw-frame messages once a new draw
  // frame has been captured, or an empty vector if no frame is ready.
  std::vector<netui::Bytes> TakeMessages();

  // The index of the last draw frame TakeMessages() returned (0 before the
  // first). The browser acks a frame by echoing its index.
  uint64_t LastFrameIndex() const;

  // ---- UI thread ------------------------------------------------------------

  // Applies queued browser input to the current Dear ImGui context. Call before
  // ImGui::NewFrame().
  void ApplyInput();

  // Records texture changes and serialized draw lists for TakeMessages().
  // Call after ImGui::Render() on the current Dear ImGui context.
  void CaptureFrame(ImGuiMouseCursor mouse_cursor);

  // Queues UTF-8 clipboard text (or an open-URL command) from Dear ImGui's
  // Platform_SetClipboardTextFn to be sent before the next draw frame.
  void QueueClipboard(std::string_view text);

 private:
  // Encodes pending Dear ImGui texture updates (or all live textures when
  // `resend_all` is true).
  std::vector<netui::Bytes> CollectTextureChanges(bool resend_all);

  // Turns one browser Input message into Dear ImGui input events.
  void ApplyOneInput(const netui::Input& input);

  // Releases every key and button the previous browser left pressed.
  void ReleaseInput();

  // Clears the per-session state. Requires mutex_.
  void ResetLocked();

  mutable std::mutex mutex_;
  // Guarded by mutex_:
  netui::SerializedFrame pending_frame_;        // Newest frame, serialized.
  netui::SerializedFrame reference_frame_;      // Last frame taken by browser.
  std::vector<netui::Bytes> pending_textures_;  // Oldest first.
  std::vector<netui::Bytes> pending_clipboards_;  // Oldest first.
  std::vector<netui::Input> pending_inputs_;      // Oldest first.
  uint64_t session_id_ = 0;        // Incremented by every StartSession().
  uint64_t next_frame_index_ = 1;  // Index of the next frame taken.
  bool active_ = false;
  bool has_input_ = false;          // The session's browser has sent input.
  bool resend_textures_ = false;    // Re-create all textures with next frame.
  bool release_input_ = false;      // Release the previous browser's input.
  bool full_frame_wanted_ = true;   // Send the next frame uncompressed.
  bool has_pending_frame_ = false;  // A new frame awaits sending.
  bool has_reference_frame_ = false;

  // UI thread only:
  std::unordered_set<int> live_textures_;  // IDs the browser was sent.
  uint64_t frame_session_id_ = 0;  // The session the frame is laid out for.
  netui::Input previous_input_;    // The last applied Input.
  int next_texture_id_ = 1;
  bool has_previous_input_ = false;
};

}  // namespace mujoco::studio

#endif  // MUJOCO_PYTHON_STUDIO_WEB_HEADLESS_UI_SESSION_H_
