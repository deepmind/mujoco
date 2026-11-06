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
// NetImgui messages for the browser (textures, clipboards, and delta-compressed
// draw frames) and converts browser NetImgui Input messages into Dear ImGui
// input events.
//
// Threading: the UI thread calls ApplyInput(), CaptureFrame(), and
// QueueClipboard(); the web server thread calls StartSession(), EndSession(),
// IsActive(), HasInput(), PushInput(), and TakeMessages(). All Dear ImGui calls
// and frame encoding happen on the UI thread.

#ifndef MUJOCO_PYTHON_STUDIO_WEB_HEADLESS_UI_SESSION_H_
#define MUJOCO_PYTHON_STUDIO_WEB_HEADLESS_UI_SESSION_H_

#include <cstddef>
#include <cstdint>
#include <memory>
#include <mutex>
#include <span>
#include <string>
#include <string_view>
#include <unordered_set>
#include <vector>

#include <imgui.h>
#include <NetImgui_CmdPackets.h>

namespace mujoco::studio {

class HeadlessUiSession {
 public:
  // `client_name` (UTF-8) is reported to the browser in the handshake,
  // truncated on a UTF-8 character boundary to fit 63 bytes.
  explicit HeadlessUiSession(std::string client_name);
  ~HeadlessUiSession();

  HeadlessUiSession(const HeadlessUiSession&) = delete;
  HeadlessUiSession& operator=(const HeadlessUiSession&) = delete;

  // ---- Server thread --------------------------------------------------------

  // Starts a session for a newly connected browser from its NetImgui Version
  // handshake message. Populates `reply` on success or `error` on failure.
  bool StartSession(std::span<const std::byte> handshake, std::string* reply,
                    std::string* error);

  // Ends the session. No frames are captured until the next StartSession().
  void EndSession();

  // True between a successful StartSession() and EndSession().
  bool IsActive() const;

  // True once the active session's browser has sent input.
  bool HasInput() const;

  // Queues one Input command from the browser for the next ApplyInput().
  // Returns false if the message is invalid or no session is active.
  bool PushInput(std::span<const std::byte> message);

  // Returns pending texture, clipboard, and draw-frame messages once a new draw
  // frame has been captured, or an empty vector if no frame is ready.
  std::vector<std::string> TakeMessages();

  // ---- UI thread ------------------------------------------------------------

  // Applies queued browser input to the current Dear ImGui context. Call before
  // ImGui::NewFrame().
  void ApplyInput();

  // Encodes texture changes and the rendered draw frame for TakeMessages().
  // Call after ImGui::Render() on the current Dear ImGui context.
  void CaptureFrame(ImGuiMouseCursor mouse_cursor);

  // Queues UTF-8 clipboard text (or an open-URL command) from Dear ImGui's
  // Platform_SetClipboardTextFn to be sent before the next draw frame.
  void QueueClipboard(std::string_view text);

 private:
  struct FrameDeleter {
    void operator()(NetImgui::Internal::CmdDrawFrame* frame) const;
  };
  using FramePtr =
      std::unique_ptr<NetImgui::Internal::CmdDrawFrame, FrameDeleter>;

  // Encodes pending Dear ImGui texture updates (or all live textures when
  // `resend_all` is true).
  std::vector<std::string> CollectTextureChanges(bool resend_all);

  // Turns one browser Input command into Dear ImGui input events.
  void ApplyOneInput(const NetImgui::Internal::CmdInput& input);

  // Releases every key and button the previous browser left pressed.
  void ReleaseInput();

  // Clears the per-session state. Requires mutex_.
  void ResetLocked();

  const std::string client_name_;

  mutable std::mutex mutex_;
  // Guarded by mutex_:
  std::string pending_frame_message_;                         // Newest frame.
  std::vector<std::string> pending_textures_;                 // Oldest first.
  std::vector<std::string> pending_clipboards_;               // Oldest first.
  std::vector<NetImgui::Internal::CmdInput> pending_inputs_;  // Oldest first.
  FramePtr pending_frame_;              // ... and as a frame.
  FramePtr reference_frame_;  // The last frame taken: the browser has it.
  uint64_t session_id_ = 0;          // Incremented by every StartSession().
  uint64_t next_frame_index_ = 1;    // Index of the next frame taken.
  bool active_ = false;
  bool has_input_ = false;           // The session's browser has sent input.
  bool resend_textures_ = false;     // Re-create all textures with next frame.
  bool release_input_ = false;       // Release the previous browser's input.
  bool full_frame_wanted_ = true;    // Send the next frame uncompressed.
  bool compression_enabled_ = true;  // The browser accepts compressed frames.
  bool pending_frame_is_full_ = false;  // ... and not compressed.

  // UI thread only:
  std::unordered_set<int> live_textures_;        // IDs the browser was sent.
  uint64_t frame_session_id_ = 0;  // The session the frame is laid out for.
  NetImgui::Internal::CmdInput previous_input_;  // The last applied Input.
  int next_texture_id_ = 1;
  bool has_previous_input_ = false;
};

}  // namespace mujoco::studio

#endif  // MUJOCO_PYTHON_STUDIO_WEB_HEADLESS_UI_SESSION_H_
