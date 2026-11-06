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

#include "headless_ui_session.h"

#include <algorithm>
#include <cfloat>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <iterator>
#include <mutex>
#include <string>
#include <string_view>
#include <unordered_set>
#include <utility>
#include <vector>

#include <imgui.h>
#include "logging.h"
#include "ui_codec.h"

namespace mujoco::studio {
namespace {

// Input messages kept while the UI thread doesn't run frames (two seconds'
// worth at 60 Hz); older ones are dropped.
constexpr size_t kMaxPendingInputs = 120;

// Clipboard messages kept between TakeMessages() calls; older ones are dropped.
constexpr size_t kMaxPendingClipboards = 16;

// Studio apps ask the browser to open a URL by setting the clipboard text to
// the URL with this prefix, since the headless server has no browser of its
// own. QueueClipboard() sends such text as a Clipboard::kOpenUrl message.
constexpr std::string_view kOpenUrlPrefix = "\x01OPEN_URL:";

// Maps a Dear ImGui texture format to the Studio UI wire texture format.
netui::Texture::Format ToTextureFormat(ImTextureFormat format) {
  return format == ImTextureFormat_Alpha8 ? netui::Texture::Format::kAlpha8
                                          : netui::Texture::Format::kRgba8;
}

// A Texture message carrying the pixels inside `rect` of `texture`.
netui::Bytes TextureMessage(ImTextureData& texture,
                            netui::Texture::Action action,
                            const ImTextureRect& rect) {
  const size_t row_size =
      static_cast<size_t>(rect.w) * static_cast<size_t>(texture.BytesPerPixel);
  const size_t pixels_size = row_size * static_cast<size_t>(rect.h);
  netui::Bytes message(sizeof(netui::Texture) + pixels_size);
  netui::Texture header;
  header.header.size = static_cast<uint32_t>(message.size());
  header.tex_id = static_cast<uint64_t>(texture.UniqueID);
  header.action = action;
  header.format = ToTextureFormat(texture.Format);
  header.x = rect.x;
  header.y = rect.y;
  header.width = rect.w;
  header.height = rect.h;
  std::memcpy(message.data(), &header, sizeof(header));

  std::byte* pixels = message.data() + sizeof(header);
  for (int y = 0; y < rect.h; ++y) {
    std::memcpy(pixels + static_cast<size_t>(y) * row_size,
                texture.GetPixelsAt(rect.x, rect.y + y), row_size);
  }
  return message;
}

// A Texture message telling the browser to destroy a texture.
netui::Bytes DestroyTextureMessage(int texture_id) {
  netui::Texture header;
  header.tex_id = static_cast<uint64_t>(texture_id);
  header.action = netui::Texture::Action::kDestroy;
  const auto* raw = reinterpret_cast<const std::byte*>(&header);
  return netui::Bytes(raw, raw + sizeof(header));
}

// Returns the bounding rectangle covering all pixels of `texture`.
ImTextureRect WholeTexture(const ImTextureData& texture) {
  return {0, 0, static_cast<uint16_t>(texture.Width),
          static_cast<uint16_t>(texture.Height)};
}

// True when `rect` spans the entire area of `texture`.
bool IsWholeTexture(const ImTextureData& texture, const ImTextureRect& rect) {
  return rect.x == 0 && rect.y == 0 && rect.w == texture.Width &&
         rect.h == texture.Height;
}

// Copies the fixed-size Texture header out of an encoded texture message.
netui::Texture ReadTextureHeader(netui::BytesView message) {
  netui::Texture header;
  std::memcpy(&header, message.data(), sizeof(header));
  return header;
}

// Turns the change from input state `before` (null: the session's first input)
// to `after` into Dear ImGui key, mouse button, wheel and character events.
void AddInputEvents(ImGuiIO& io, const netui::Input* before,
                    const netui::Input& after) {
  for (int i = 0; i < ImGuiKey_NamedKey_COUNT; ++i) {
    const auto key = static_cast<ImGuiKey>(ImGuiKey_NamedKey_BEGIN + i);
    // Dear ImGui derives the mouse "keys" from mouse button events, and takes
    // the modifiers as ImGuiMod_ flags (below).
    if ((key >= ImGuiKey_MouseLeft && key <= ImGuiKey_MouseWheelY) ||
        (key >= ImGuiKey_ReservedForModCtrl &&
         key <= ImGuiKey_ReservedForModSuper)) {
      continue;
    }
    const bool down = after.IsKeyDown(i);
    const bool was_down = before != nullptr && before->IsKeyDown(i);
    if (down != was_down) {
      io.AddKeyEvent(key, down);
    }
  }

  constexpr struct {
    ImGuiKey key;
    ImGuiKey mod;
  } kModifiers[] = {
      {ImGuiKey_ReservedForModCtrl, ImGuiMod_Ctrl},
      {ImGuiKey_ReservedForModShift, ImGuiMod_Shift},
      {ImGuiKey_ReservedForModAlt, ImGuiMod_Alt},
      {ImGuiKey_ReservedForModSuper, ImGuiMod_Super},
  };
  for (const auto& modifier : kModifiers) {
    const bool down = after.IsImGuiKeyDown(modifier.key);
    const bool was_down =
        before != nullptr && before->IsImGuiKeyDown(modifier.key);
    if (down != was_down) {
      io.AddKeyEvent(modifier.mod, down);
    }
  }

  for (int button = 0; button < ImGuiMouseButton_COUNT; ++button) {
    const uint32_t mask = 1u << button;
    const bool down = (after.mouse_down_mask & mask) != 0;
    const bool was_down =
        before != nullptr && (before->mouse_down_mask & mask) != 0;
    if (down != was_down) {
      io.AddMouseButtonEvent(button, down);
    }
  }

  // The wheel fields are running totals; each event is the change since the
  // previous input. A session's first input only sets the starting totals.
  if (before != nullptr) {
    const float wheel_x = after.mouse_wheel[1] - before->mouse_wheel[1];
    const float wheel_y = after.mouse_wheel[0] - before->mouse_wheel[0];
    if (wheel_x != 0.0f || wheel_y != 0.0f) {
      io.AddMouseWheelEvent(wheel_x, wheel_y);
    }
  }

  for (uint32_t i = 0; i < after.char_count; ++i) {
    io.AddInputCharacter(after.chars[i]);
  }
}

}  // namespace

HeadlessUiSession::~HeadlessUiSession() = default;

bool HeadlessUiSession::StartSession(netui::BytesView handshake,
                                     netui::Bytes* reply, std::string* error) {
  netui::Hello ours;
  netui::Hello theirs;
  if (handshake.size() != sizeof(netui::Hello)) {
    *error = "the handshake has " + std::to_string(handshake.size()) +
             " bytes, expected " + std::to_string(sizeof(netui::Hello));
    return false;
  }
  std::memcpy(&theirs, handshake.data(), sizeof(theirs));
  if (theirs.header.kind != netui::MessageKind::kHello ||
      theirs.header.size != sizeof(netui::Hello)) {
    *error = "the handshake is not a Studio UI Hello message";
    return false;
  }
  if (theirs.header.version != ours.header.version) {
    *error = "browser uses Studio UI protocol version " +
             std::to_string(static_cast<int>(theirs.header.version)) +
             ", viewer uses version " +
             std::to_string(static_cast<int>(ours.header.version)) +
             " (reload the page)";
    return false;
  }
  const auto* raw = reinterpret_cast<const std::byte*>(&ours);
  reply->assign(raw, raw + sizeof(ours));

  std::lock_guard<std::mutex> lock(mutex_);
  ResetLocked();
  active_ = true;
  ++session_id_;
  resend_textures_ = true;
  LOG(Info, "UI session %llu started",
      static_cast<unsigned long long>(session_id_));
  return true;
}

void HeadlessUiSession::EndSession() {
  std::lock_guard<std::mutex> lock(mutex_);
  if (!active_) return;
  ResetLocked();
  active_ = false;
  LOG(Info, "UI session %llu ended",
      static_cast<unsigned long long>(session_id_));
}

void HeadlessUiSession::ResetLocked() {
  has_input_ = false;
  release_input_ = true;
  full_frame_wanted_ = true;
  pending_textures_.clear();
  pending_clipboards_.clear();
  pending_inputs_.clear();
  has_pending_frame_ = false;
  pending_frame_.Clear();
  has_reference_frame_ = false;
  reference_frame_.Clear();
}

bool HeadlessUiSession::IsActive() const {
  std::lock_guard<std::mutex> lock(mutex_);
  return active_;
}

bool HeadlessUiSession::HasInput() const {
  std::lock_guard<std::mutex> lock(mutex_);
  return active_ && has_input_;
}

bool HeadlessUiSession::PushInput(netui::BytesView message) {
  netui::Input input;
  if (!netui::DecodeInput(message, &input)) {
    return false;
  }

  std::lock_guard<std::mutex> lock(mutex_);
  if (!active_) return false;
  // The browser asks for a full frame after connecting, and whenever it gets a
  // compressed frame it cannot decompress.
  if ((input.header.flags & netui::Input::kRequestKeyframe) != 0) {
    full_frame_wanted_ = true;
  }
  if (pending_inputs_.size() >= kMaxPendingInputs) {
    pending_inputs_.erase(pending_inputs_.begin());
  }
  pending_inputs_.push_back(input);
  has_input_ = true;
  return true;
}

std::vector<netui::Bytes> HeadlessUiSession::TakeMessages() {
  std::lock_guard<std::mutex> lock(mutex_);
  std::vector<netui::Bytes> messages;
  // Texture and clipboard updates wait until a draw frame is ready so every
  // non-empty batch ends with a draw frame (the message the browser acks).
  // If multiple frames were captured since the last TakeMessages() call, only
  // the newest frame is encoded here against `reference_frame_` (the last frame
  // returned here) unless a full frame was requested.
  if (!active_ || !has_pending_frame_) return messages;
  const bool send_full = full_frame_wanted_ || !has_reference_frame_;
  netui::Bytes frame_message =
      netui::EncodeDrawFrame(pending_frame_, next_frame_index_,
                             send_full ? nullptr : &reference_frame_);
  has_pending_frame_ = false;
  if (frame_message.empty()) {
    // The browser would reject a frame this large, ask for a keyframe, and get
    // the same frame again, so drop it and say why. Texture and clipboard
    // updates wait for the next frame.
    LOG(Error, "TakeMessages: dropping a draw frame larger than %zu bytes",
        netui::kMaxMessageBytes);
    return messages;
  }
  messages.swap(pending_textures_);
  messages.insert(messages.end(),
                  std::make_move_iterator(pending_clipboards_.begin()),
                  std::make_move_iterator(pending_clipboards_.end()));
  pending_clipboards_.clear();
  messages.push_back(std::move(frame_message));
  if (send_full) full_frame_wanted_ = false;
  // The browser will have this frame: the next one is compressed against it.
  reference_frame_ = std::move(pending_frame_);
  has_reference_frame_ = true;
  ++next_frame_index_;
  return messages;
}

uint64_t HeadlessUiSession::LastFrameIndex() const {
  std::lock_guard<std::mutex> lock(mutex_);
  return next_frame_index_ - 1;
}

void HeadlessUiSession::ApplyInput() {
  std::vector<netui::Input> inputs;
  bool release = false;
  {
    std::lock_guard<std::mutex> lock(mutex_);
    inputs.swap(pending_inputs_);
    release = release_input_;
    release_input_ = false;
    // Record which session this frame is laid out for once its browser has sent
    // input (its screen size). CaptureFrame() and QueueClipboard() drop work if
    // the session changes or ends before ImGui::Render() finishes. Session IDs
    // start at 1, so 0 stands for no session.
    frame_session_id_ = (active_ && has_input_) ? session_id_ : 0;
  }
  if (release) ReleaseInput();
  // Every input is applied in order (Dear ImGui queues the events), so a click
  // shorter than a frame is not lost.
  for (const netui::Input& input : inputs) {
    ApplyOneInput(input);
  }
}

void HeadlessUiSession::ApplyOneInput(const netui::Input& input) {
  ImGuiIO& io = ImGui::GetIO();
  if (input.screen_size[0] > 0 && input.screen_size[1] > 0) {
    io.DisplaySize = ImVec2(input.screen_size[0], input.screen_size[1]);
  }
  if (std::isfinite(input.framebuffer_scale) &&
      input.framebuffer_scale > 0.0f) {
    const float scale =
        std::round(std::clamp(input.framebuffer_scale, 0.5f, 4.0f) * 100.0f) *
        0.01f;
    io.DisplayFramebufferScale = ImVec2(scale, scale);
  }
  const bool macos = (input.header.flags & netui::Input::kMacOsBehaviors) != 0;
  if (io.ConfigMacOSXBehaviors != macos) {
    if (has_previous_input_) {
      ReleaseInput();
    }
    io.ConfigMacOSXBehaviors = macos;
  }
  io.AddMousePosEvent(input.mouse_pos[0], input.mouse_pos[1]);
  AddInputEvents(io, has_previous_input_ ? &previous_input_ : nullptr, input);
  previous_input_ = input;
  has_previous_input_ = true;
}

void HeadlessUiSession::ReleaseInput() {
  if (!has_previous_input_) return;
  // Diffing the previous input state against a default-constructed Input (with
  // the running mouse-wheel totals preserved so no scroll delta is emitted)
  // releases every key and mouse button held by the previous browser.
  netui::Input released;
  released.mouse_wheel[0] = previous_input_.mouse_wheel[0];
  released.mouse_wheel[1] = previous_input_.mouse_wheel[1];
  ImGuiIO& io = ImGui::GetIO();
  AddInputEvents(io, &previous_input_, released);
  // Mark the mouse cursor as outside the viewport so hovered widgets clear.
  io.AddMousePosEvent(-FLT_MAX, -FLT_MAX);
  has_previous_input_ = false;
}

void HeadlessUiSession::CaptureFrame(ImGuiMouseCursor mouse_cursor) {
  bool resend_all = false;
  {
    std::lock_guard<std::mutex> lock(mutex_);
    // A frame laid out before the session's browser sent input, or for the
    // previous browser (control changed hands during the frame), is not sent.
    // The session's first frame then re-sends everything.
    if (!active_ || session_id_ != frame_session_id_) return;
    resend_all = resend_textures_;
    resend_textures_ = false;
  }

  // Textures first: the draw frame refers to them by the IDs assigned here.
  std::vector<netui::Bytes> textures = CollectTextureChanges(resend_all);
  netui::SerializedFrame frame;
  netui::SerializeDrawData(ImGui::GetDrawData(), mouse_cursor, &frame);

  std::lock_guard<std::mutex> lock(mutex_);
  if (!active_ || session_id_ != frame_session_id_) {
    // The session changed meanwhile; the new one's first frame re-sends all.
    return;
  }
  for (netui::Bytes& message : textures) {
    const netui::Texture added = ReadTextureHeader(message);
    if (added.action != netui::Texture::Action::kUpdate) {
      // A Create replaces the browser's copy of the texture and a Destroy
      // removes it, so any earlier changes still queued for the same texture ID
      // before TakeMessages() are stale and can be dropped.
      pending_textures_.erase(
          std::remove_if(pending_textures_.begin(), pending_textures_.end(),
                         [&added](const netui::Bytes& queued) {
                           return ReadTextureHeader(queued).tex_id ==
                                  added.tex_id;
                         }),
          pending_textures_.end());
    }
    pending_textures_.push_back(std::move(message));
  }

  // Store the serialized draw lists; TakeMessages() encodes only the newest
  // frame that actually goes out on the wire, replacing any earlier frame
  // captured since the last TakeMessages() call.
  pending_frame_ = std::move(frame);
  has_pending_frame_ = true;
}

void HeadlessUiSession::QueueClipboard(std::string_view text) {
  const bool open_url = text.starts_with(kOpenUrlPrefix);
  if (open_url) text.remove_prefix(kOpenUrlPrefix.size());
  netui::Bytes message = netui::EncodeClipboard(text, open_url);
  std::lock_guard<std::mutex> lock(mutex_);
  if (!active_ || session_id_ != frame_session_id_) return;
  if (pending_clipboards_.size() >= kMaxPendingClipboards) {
    pending_clipboards_.erase(pending_clipboards_.begin());
  }
  pending_clipboards_.push_back(std::move(message));
}

std::vector<netui::Bytes> HeadlessUiSession::CollectTextureChanges(
    bool resend_all) {
  // There is no renderer here: this plays the part of a Dear ImGui renderer
  // backend (see ImGuiBackendFlags_RendererHasTextures), except that it turns
  // each texture request into a message for the browser.
  std::vector<netui::Bytes> messages;
  std::unordered_set<int> present;
  for (ImTextureData* texture : ImGui::GetPlatformIO().Textures) {
    switch (texture->Status) {
      case ImTextureStatus_WantCreate:
        // The draw frame encoder identifies textures by UniqueID, which Dear
        // ImGui only numbers per font atlas: number them across all textures.
        texture->UniqueID = next_texture_id_++;
        live_textures_.insert(texture->UniqueID);
        messages.push_back(TextureMessage(
            *texture, netui::Texture::Action::kCreate, WholeTexture(*texture)));
        texture->SetTexID(static_cast<ImTextureID>(texture->UniqueID));
        texture->SetStatus(ImTextureStatus_OK);
        break;
      case ImTextureStatus_WantUpdates:
        // A whole-texture update (e.g. an image re-uploaded from Python) goes
        // as a Create, which the browser treats as a replacement: it then
        // supersedes the changes still queued for the texture.
        if (resend_all || IsWholeTexture(*texture, texture->UpdateRect)) {
          messages.push_back(TextureMessage(*texture,
                                            netui::Texture::Action::kCreate,
                                            WholeTexture(*texture)));
        } else if (texture->UpdateRect.w > 0 && texture->UpdateRect.h > 0) {
          messages.push_back(TextureMessage(
              *texture, netui::Texture::Action::kUpdate, texture->UpdateRect));
        }
        texture->SetStatus(ImTextureStatus_OK);
        break;
      case ImTextureStatus_WantDestroy:
        // Wait until no frame uses the texture any more.
        if (texture->UnusedFrames > 0) {
          messages.push_back(DestroyTextureMessage(texture->UniqueID));
          live_textures_.erase(texture->UniqueID);
          texture->SetTexID(ImTextureID_Invalid);
          texture->SetStatus(ImTextureStatus_Destroyed);
          continue;
        }
        if (resend_all) {
          messages.push_back(TextureMessage(*texture,
                                            netui::Texture::Action::kCreate,
                                            WholeTexture(*texture)));
        }
        break;
      case ImTextureStatus_OK:
        if (resend_all) {
          messages.push_back(TextureMessage(*texture,
                                            netui::Texture::Action::kCreate,
                                            WholeTexture(*texture)));
        }
        break;
      case ImTextureStatus_Destroyed:
        continue;
    }
    present.insert(texture->UniqueID);
  }

  // Textures removed without a destroy request (e.g. unregistered user
  // textures) are destroyed in the browser too.
  for (auto it = live_textures_.begin(); it != live_textures_.end();) {
    if (present.count(*it) == 0) {
      messages.push_back(DestroyTextureMessage(*it));
      it = live_textures_.erase(it);
    } else {
      ++it;
    }
  }
  return messages;
}

}  // namespace mujoco::studio
