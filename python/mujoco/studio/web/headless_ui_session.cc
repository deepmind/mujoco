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
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <iterator>
#include <mutex>
#include <new>
#include <span>
#include <string>
#include <string_view>
#include <unordered_set>
#include <utility>
#include <vector>

#include <imgui.h>
#include <NetImgui_CmdPackets.h>
#include "google/logging.h"

namespace mujoco::studio {
namespace {

using ::NetImgui::Internal::CmdClipboard;
using ::NetImgui::Internal::CmdDrawFrame;
using ::NetImgui::Internal::CmdHeader;
using ::NetImgui::Internal::CmdInput;
using ::NetImgui::Internal::CmdTexture;
using ::NetImgui::Internal::CmdVersion;
using ::NetImgui::Internal::CompressCmdDrawFrame;
using ::NetImgui::Internal::ConvertTextureFormat;
using ::NetImgui::Internal::ConvertToCmdDrawFrame;
using ::NetImgui::Internal::DivUp;
using ::NetImgui::Internal::netImguiDelete;
using ::NetImgui::Internal::StringCopy;

// The browser describes keys with NetImgui's copy of Dear ImGui's named keys,
// which lists them in the same order. These catch an ImGui update that changes
// the order (NetImgui's copy then needs the same update).
static_assert(static_cast<int>(CmdInput::ImGuiKey_COUNT) ==
              ImGuiKey_NamedKey_COUNT);
static_assert(static_cast<int>(CmdInput::ImGuiKey_Tab) ==
              ImGuiKey_Tab - ImGuiKey_NamedKey_BEGIN);
static_assert(static_cast<int>(CmdInput::ImGuiKey_MouseLeft) ==
              ImGuiKey_MouseLeft - ImGuiKey_NamedKey_BEGIN);
static_assert(static_cast<int>(CmdInput::ImGuiKey_ReservedForModCtrl) ==
              ImGuiKey_ReservedForModCtrl - ImGuiKey_NamedKey_BEGIN);
static_assert(static_cast<int>(CmdInput::ImGuiMouseButton_COUNT) ==
              ImGuiMouseButton_COUNT);

// Input commands kept while the UI thread doesn't run frames (two seconds'
// worth at 60 Hz); older ones are dropped.
constexpr size_t kMaxPendingInputs = 120;

// Clipboard messages kept between TakeMessages() calls; older ones are dropped.
constexpr size_t kMaxPendingClipboards = 16;
constexpr size_t kMaxClipboardBytes = 1024 * 1024;

// Commands are built in 8-byte aligned memory, as NetImgui expects.
template <typename TCommand>
TCommand* NewCommand(std::vector<uint64_t>* buffer, size_t size) {
  buffer->assign(DivUp<size_t>(size, sizeof(uint64_t)), 0);
  auto* command = new (buffer->data()) TCommand();
  command->mSize = static_cast<uint32_t>(size);
  return command;
}

// One NetImgui command (in its sendable, offset form) as one message.
std::string ToMessage(const CmdHeader& command) {
  return std::string(reinterpret_cast<const char*>(&command), command.mSize);
}

// A Texture command carrying the pixels inside `rect` of `texture`.
std::string TextureMessage(ImTextureData& texture, CmdTexture::eType status,
                           const ImTextureRect& rect) {
  const size_t row_size = static_cast<size_t>(rect.w) * texture.BytesPerPixel;
  std::vector<uint64_t> buffer;
  auto* command =
      NewCommand<CmdTexture>(&buffer, sizeof(CmdTexture) + row_size * rect.h);
  command->mTextureClientID = static_cast<uint64_t>(texture.UniqueID);
  command->mStatus = status;
  command->mFormat = static_cast<uint8_t>(ConvertTextureFormat(texture.Format));
  command->mUpdatable = true;
  command->mIsDearImGuiManaged = true;
  command->mWidth = rect.w;
  command->mHeight = rect.h;
  command->mOffsetX = rect.x;
  command->mOffsetY = rect.y;
  auto* pixels = reinterpret_cast<uint8_t*>(command + 1);
  for (int y = 0; y < rect.h; ++y) {
    std::memcpy(pixels + y * row_size, texture.GetPixelsAt(rect.x, rect.y + y),
                row_size);
  }
  command->mpTextureData.SetPtr(pixels);
  command->mpTextureData.ToOffset();
  return ToMessage(*command);
}

// A Texture command telling the browser to destroy a texture.
std::string DestroyTextureMessage(int texture_id) {
  std::vector<uint64_t> buffer;
  auto* command = NewCommand<CmdTexture>(&buffer, sizeof(CmdTexture));
  command->mTextureClientID = static_cast<uint64_t>(texture_id);
  command->mStatus = CmdTexture::eType::Destroy;
  command->mpTextureData.SetPtr(reinterpret_cast<uint8_t*>(command + 1));
  command->mpTextureData.ToOffset();
  return ToMessage(*command);
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

// The fixed part of a Texture message built above; its pixels follow.
CmdTexture TextureHeader(std::string_view message) {
  CmdTexture header;
  std::memcpy(static_cast<void*>(&header), message.data(), sizeof(header));
  return header;
}

// Truncates UTF-8 `text` to at most `max_size` bytes on a character boundary.
std::string TruncateUtf8(std::string text, size_t max_size) {
  if (text.size() > max_size) {
    size_t end = max_size;
    // Back up over the continuation bytes (10xxxxxx) of a character that would
    // be cut, to its first byte.
    while (end > 0 && (static_cast<unsigned char>(text[end]) & 0xC0) == 0x80) {
      --end;
    }
    text.resize(end);
  }
  return text;
}

// A Clipboard command carrying NUL-terminated UTF-8 `text`.
std::string ClipboardMessage(std::string_view text) {
  const std::string utf8 = TruncateUtf8(std::string(text), kMaxClipboardBytes);
  const size_t byte_size = utf8.size() + 1;
  std::vector<uint64_t> buffer;
  auto* command =
      NewCommand<CmdClipboard>(&buffer, sizeof(CmdClipboard) + byte_size);
  command->mByteSize = byte_size;
  auto* content = reinterpret_cast<char*>(command + 1);
  std::memcpy(content, utf8.data(), utf8.size());
  content[utf8.size()] = '\0';
  command->mContentUTF8.SetPtr(content);
  command->mContentUTF8.ToOffset();
  return ToMessage(*command);
}

// Turns the change from input state `before` (null: the session's first input)
// to `after` into Dear ImGui key, mouse button, wheel and character events.
void AddInputEvents(ImGuiIO& io, const CmdInput* before,
                    const CmdInput& after) {
  for (int i = 0; i < CmdInput::ImGuiKey_COUNT; ++i) {
    const auto key = static_cast<ImGuiKey>(ImGuiKey_NamedKey_BEGIN + i);
    // Dear ImGui derives the mouse "keys" from mouse button events, and takes
    // the modifiers as ImGuiMod_ flags (below).
    if ((key >= ImGuiKey_MouseLeft && key <= ImGuiKey_MouseWheelY) ||
        (key >= ImGuiKey_ReservedForModCtrl &&
         key <= ImGuiKey_ReservedForModSuper)) {
      continue;
    }
    const auto netimgui_key = static_cast<CmdInput::NetImguiKeys>(i);
    const bool down = after.IsKeyDown(netimgui_key);
    const bool was_down = before != nullptr && before->IsKeyDown(netimgui_key);
    if (down != was_down) {
      io.AddKeyEvent(key, down);
    }
  }

  constexpr struct {
    CmdInput::NetImguiKeys key;
    ImGuiKey mod;
  } kModifiers[] = {
      {CmdInput::ImGuiKey_ReservedForModCtrl, ImGuiMod_Ctrl},
      {CmdInput::ImGuiKey_ReservedForModShift, ImGuiMod_Shift},
      {CmdInput::ImGuiKey_ReservedForModAlt, ImGuiMod_Alt},
      {CmdInput::ImGuiKey_ReservedForModSuper, ImGuiMod_Super},
  };
  for (const auto& modifier : kModifiers) {
    const bool down = after.IsKeyDown(modifier.key);
    const bool was_down = before != nullptr && before->IsKeyDown(modifier.key);
    if (down != was_down) {
      io.AddKeyEvent(modifier.mod, down);
    }
  }

  for (int button = 0; button < CmdInput::ImGuiMouseButton_COUNT; ++button) {
    const uint64_t mask = uint64_t{1} << button;
    const bool down = (after.mMouseDownMask & mask) != 0;
    const bool was_down =
        before != nullptr && (before->mMouseDownMask & mask) != 0;
    if (down != was_down) {
      io.AddMouseButtonEvent(button, down);
    }
  }

  // The wheel fields are running totals; each event is the change since the
  // previous input. A session's first input only sets the starting totals.
  if (before != nullptr) {
    const float wheel_x = after.mMouseWheelHoriz - before->mMouseWheelHoriz;
    const float wheel_y = after.mMouseWheelVert - before->mMouseWheelVert;
    if (wheel_x != 0.0f || wheel_y != 0.0f) {
      io.AddMouseWheelEvent(wheel_x, wheel_y);
    }
  }

  for (uint16_t i = 0; i < after.mKeyCharCount; ++i) {
    io.AddInputCharacterUTF16(after.mKeyChars[i]);
  }
}

}  // namespace

void HeadlessUiSession::FrameDeleter::operator()(CmdDrawFrame* frame) const {
  netImguiDelete(frame);
}

HeadlessUiSession::HeadlessUiSession(std::string client_name)
    // The handshake's name field holds 63 bytes and a terminating NUL.
    : client_name_(TruncateUtf8(std::move(client_name),
                                sizeof(CmdVersion::mClientName) - 1)) {}

HeadlessUiSession::~HeadlessUiSession() = default;

bool HeadlessUiSession::StartSession(std::span<const std::byte> handshake,
                                     std::string* reply, std::string* error) {
  CmdVersion ours;
  CmdVersion theirs;
  if (handshake.size() != sizeof(CmdVersion)) {
    *error = "the handshake has " + std::to_string(handshake.size()) +
             " bytes, expected " + std::to_string(sizeof(CmdVersion));
    return false;
  }
  std::memcpy(static_cast<void*>(&theirs), handshake.data(), handshake.size());
  if (theirs.mType != CmdHeader::eCommands::Version ||
      theirs.mSize != sizeof(CmdVersion)) {
    *error = "the handshake is not a NetImgui Version command";
    return false;
  }
  if (theirs.mVersion != ours.mVersion) {
    *error = "browser uses NetImgui protocol version " +
             std::to_string(static_cast<int>(theirs.mVersion)) +
             ", viewer uses version " +
             std::to_string(static_cast<int>(ours.mVersion)) +
             " (reload the page)";
    return false;
  }
  if (theirs.mWCharSize != ours.mWCharSize) {
    *error = "the browser's ImWchar has " +
             std::to_string(static_cast<int>(theirs.mWCharSize)) +
             " bytes, this viewer's has " +
             std::to_string(static_cast<int>(ours.mWCharSize));
    return false;
  }
  StringCopy(ours.mClientName, client_name_.c_str());
  reply->assign(reinterpret_cast<const char*>(&ours), sizeof(ours));

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
  compression_enabled_ = true;
  pending_textures_.clear();
  pending_clipboards_.clear();
  pending_inputs_.clear();
  pending_frame_message_.clear();
  pending_frame_.reset();
  reference_frame_.reset();
}

bool HeadlessUiSession::IsActive() const {
  std::lock_guard<std::mutex> lock(mutex_);
  return active_;
}

bool HeadlessUiSession::HasInput() const {
  std::lock_guard<std::mutex> lock(mutex_);
  return active_ && has_input_;
}

bool HeadlessUiSession::PushInput(std::span<const std::byte> message) {
  if (message.size() != sizeof(CmdInput)) return false;
  CmdInput input;
  std::memcpy(static_cast<void*>(&input), message.data(), message.size());
  if (input.mType != CmdHeader::eCommands::Input ||
      input.mSize != sizeof(CmdInput) ||
      input.mKeyCharCount > std::size(input.mKeyChars)) {
    return false;
  }

  std::lock_guard<std::mutex> lock(mutex_);
  if (!active_) return false;
  // The browser asks for a full frame after connecting, and whenever it gets a
  // compressed frame it cannot decompress.
  if (input.mCompressionSkip) full_frame_wanted_ = true;
  compression_enabled_ = input.mCompressionUse;
  if (pending_inputs_.size() >= kMaxPendingInputs) {
    pending_inputs_.erase(pending_inputs_.begin());
  }
  pending_inputs_.push_back(input);
  has_input_ = true;
  return true;
}

std::vector<std::string> HeadlessUiSession::TakeMessages() {
  std::lock_guard<std::mutex> lock(mutex_);
  std::vector<std::string> messages;
  // Texture and clipboard updates wait until a draw frame is ready so every
  // non-empty batch ends with a draw frame (the message the browser acks).
  // If multiple frames were captured since the last TakeMessages() call, only
  // the newest frame is kept, encoded against `reference_frame_` (the last
  // frame returned here) unless a full frame was requested.
  if (!active_ || pending_frame_message_.empty()) return messages;
  messages.swap(pending_textures_);
  messages.insert(messages.end(),
                  std::make_move_iterator(pending_clipboards_.begin()),
                  std::make_move_iterator(pending_clipboards_.end()));
  pending_clipboards_.clear();
  messages.push_back(std::move(pending_frame_message_));
  pending_frame_message_.clear();
  if (pending_frame_is_full_) full_frame_wanted_ = false;
  // The browser will have this frame: the next one is compressed against it.
  reference_frame_ = std::move(pending_frame_);
  ++next_frame_index_;
  return messages;
}

void HeadlessUiSession::ApplyInput() {
  std::vector<CmdInput> inputs;
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
  for (const CmdInput& input : inputs) {
    ApplyOneInput(input);
  }
}

void HeadlessUiSession::ApplyOneInput(const CmdInput& input) {
  ImGuiIO& io = ImGui::GetIO();
  if (input.mScreenSize[0] > 0 && input.mScreenSize[1] > 0) {
    io.DisplaySize = ImVec2(input.mScreenSize[0], input.mScreenSize[1]);
  }
  io.AddMousePosEvent(input.mMousePos[0], input.mMousePos[1]);
  AddInputEvents(io, has_previous_input_ ? &previous_input_ : nullptr, input);
  previous_input_ = input;
  has_previous_input_ = true;
}

void HeadlessUiSession::ReleaseInput() {
  if (!has_previous_input_) return;
  // Diffing the previous input state against a default-constructed CmdInput
  // (with the running mouse-wheel totals preserved so no scroll delta is
  // emitted) releases every key and mouse button held by the previous browser.
  CmdInput released;
  released.mMouseWheelVert = previous_input_.mMouseWheelVert;
  released.mMouseWheelHoriz = previous_input_.mMouseWheelHoriz;
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
  std::vector<std::string> textures = CollectTextureChanges(resend_all);
  FramePtr frame(ConvertToCmdDrawFrame(ImGui::GetDrawData(), mouse_cursor));

  std::lock_guard<std::mutex> lock(mutex_);
  if (!active_ || session_id_ != frame_session_id_) {
    // The session changed meanwhile; the new one's first frame re-sends all.
    return;
  }
  for (std::string& message : textures) {
    const CmdTexture added = TextureHeader(message);
    if (added.mStatus != CmdTexture::eType::Update) {
      // A Create replaces the browser's copy of the texture and a Destroy
      // removes it, so any earlier changes still queued for the same texture ID
      // before TakeMessages() are stale and can be dropped.
      pending_textures_.erase(
          std::remove_if(pending_textures_.begin(), pending_textures_.end(),
                         [&added](const std::string& queued) {
                           return TextureHeader(queued).mTextureClientID ==
                                  added.mTextureClientID;
                         }),
          pending_textures_.end());
    }
    pending_textures_.push_back(std::move(message));
  }

  // Encode the frame against `reference_frame_` (the last frame returned by
  // TakeMessages(), which the browser already has). If a previous frame was
  // captured since the last TakeMessages() call, this overwrites
  // `pending_frame_message_` so the superseded frame is never sent.
  frame->mFrameIndex = next_frame_index_;
  pending_frame_is_full_ = !compression_enabled_ || full_frame_wanted_ ||
                           reference_frame_ == nullptr;
  if (pending_frame_is_full_) {
    frame->ToOffsets();
    pending_frame_message_ = ToMessage(*frame);
    if (!frame->ToPointers()) {
      // Cannot happen for a frame built here.
      LOG(Error, "CaptureFrame: invalid draw frame");
      pending_frame_message_.clear();
      pending_frame_.reset();
      return;
    }
  } else {
    FramePtr packed(CompressCmdDrawFrame(reference_frame_.get(), frame.get()));
    packed->ToOffsets();
    pending_frame_message_ = ToMessage(*packed);
  }
  pending_frame_ = std::move(frame);
}

void HeadlessUiSession::QueueClipboard(std::string_view text) {
  std::string message = ClipboardMessage(text);
  std::lock_guard<std::mutex> lock(mutex_);
  if (!active_ || session_id_ != frame_session_id_) return;
  if (pending_clipboards_.size() >= kMaxPendingClipboards) {
    pending_clipboards_.erase(pending_clipboards_.begin());
  }
  pending_clipboards_.push_back(std::move(message));
}

std::vector<std::string> HeadlessUiSession::CollectTextureChanges(
    bool resend_all) {
  // There is no renderer here: this plays the part of a Dear ImGui renderer
  // backend (see ImGuiBackendFlags_RendererHasTextures), except that it turns
  // each texture request into a message for the browser.
  std::vector<std::string> messages;
  std::unordered_set<int> present;
  for (ImTextureData* texture : ImGui::GetPlatformIO().Textures) {
    switch (texture->Status) {
      case ImTextureStatus_WantCreate:
        // The draw frame encoder identifies textures by UniqueID, which Dear
        // ImGui only numbers per font atlas: number them across all textures.
        texture->UniqueID = next_texture_id_++;
        live_textures_.insert(texture->UniqueID);
        messages.push_back(TextureMessage(*texture, CmdTexture::eType::Create,
                                          WholeTexture(*texture)));
        texture->SetTexID(static_cast<ImTextureID>(texture->UniqueID));
        texture->SetStatus(ImTextureStatus_OK);
        break;
      case ImTextureStatus_WantUpdates:
        // A whole-texture update (e.g. an image re-uploaded from Python) goes
        // as a Create, which the browser treats as a replacement: it then
        // supersedes the changes still queued for the texture.
        if (resend_all || IsWholeTexture(*texture, texture->UpdateRect)) {
          messages.push_back(TextureMessage(*texture, CmdTexture::eType::Create,
                                            WholeTexture(*texture)));
        } else if (texture->UpdateRect.w > 0 && texture->UpdateRect.h > 0) {
          messages.push_back(TextureMessage(*texture, CmdTexture::eType::Update,
                                            texture->UpdateRect));
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
          messages.push_back(TextureMessage(*texture, CmdTexture::eType::Create,
                                            WholeTexture(*texture)));
        }
        break;
      case ImTextureStatus_OK:
        if (resend_all) {
          messages.push_back(TextureMessage(*texture, CmdTexture::eType::Create,
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
