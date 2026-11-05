// Google modifications:
//  - CreateTexture_Default copies the texture size worth of pixels, instead
//    of the data size received from the network.
#include "NetImguiServer_App.h"

#include "Fonts/Roboto_Medium.cpp"
#include "NetImguiServer_Config.h"
#include "NetImguiServer_Network.h"
#include "NetImguiServer_RemoteClient.h"
#include "NetImguiServer_UI.h"

namespace NetImguiServer {
namespace App {

constexpr uint32_t kClientCountMax =
    32;  //! @sammyfreg todo: support unlimited client count
ImVector<ServerTexture*>
    gServerTextures;  // List of ALL server created textures (used by server and
                      // clients)
ServerTexture* gServerTextureEmpty =
    nullptr;  // Empty texture used when no valid texture found
bool gLoadedConfigOnce = false;

void UpdateServerTextures();

bool Startup(const char* CmdLine) {
  //---------------------------------------------------------------------------------------------
  // Load Settings savefile and parse for auto connect commandline option
  //---------------------------------------------------------------------------------------------
  if (!gLoadedConfigOnce) {
    NetImguiServer::Config::Client::LoadAll();
    gLoadedConfigOnce = true;
  }
  AddTransientClientConfigFromString(CmdLine);

  //---------------------------------------------------------------------------------------------
  // Perform application initialization:
  //---------------------------------------------------------------------------------------------
  if (RemoteClient::Client::Startup(kClientCountMax) &&
      NetImguiServer::Network::Startup() && NetImguiServer::UI::Startup()) {
    NetImgui::Internal::CmdTexture cmdTexture;
    uint32_t EmptyData[4 * 4] = {
        0xFF0000FF, 0xFF0000FF, 0xFF0000FF, 0xFF0000FF, 0xFF0000FF, 0xFF0000FF,
        0xFF0000FF, 0xFF0000FF, 0xFF0000FF, 0xFF0000FF, 0xFF0000FF, 0xFF0000FF,
        0xFF0000FF, 0xFF0000FF, 0xFF0000FF, 0xFF0000FF};
    cmdTexture.mTextureClientID = 0;
    cmdTexture.mFormat = ImTextureFormat::ImTextureFormat_RGBA32;
    cmdTexture.mWidth = cmdTexture.mHeight = 4;
    cmdTexture.mpTextureData.SetPtr((uint8_t*)EmptyData);
    gServerTextureEmpty = CreateTexture(cmdTexture, sizeof(EmptyData));

    LoadFonts();
    ImGui::GetIO().IniFilename =
        nullptr;  // Disable server ImGui ini settings (not really needed, and
                  // avoid imgui.ini filename conflicts)
    ImGui::GetIO().LogFilename = nullptr;
    return HAL_Startup(CmdLine);
  }
  return false;
}

void Shutdown() {
  // Mark all texture resources as wanting deletion
  for (ServerTexture* texServer : gServerTextures) {
    if (texServer && texServer->IsValid()) {
      if (!texServer->mIsCustom) {
        texServer->mTexData.SetStatus(ImTextureStatus_WantDestroy);
        texServer->mTexData.UnusedFrames = 1;
      } else {
#if TEXTURE_CUSTOM_SAMPLE
        if (texServer && texServer->mIsCustom) {
          0  // TODO
        }
#endif
      }
    }
  }

  // Allow Dear ImGui to delete the textures
  ImGui::NewFrame();
  ImGui::Render();
  HAL_RenderDrawData(ImGui::GetDrawData());
  if (ImGui::GetIO().ConfigFlags & ImGuiConfigFlags_ViewportsEnable) {
    ImGui::UpdatePlatformWindows();
    ImGui::RenderPlatformWindowsDefault();
  }

  // Finish removing them from Dear ImGui user textures and delete objects
  UpdateServerTextures();

  // Update Dear ImGui texture list (so it doesn't have deleted items in it)
  ImGui::NewFrame();
  ImGui::Render();
  if (ImGui::GetIO().ConfigFlags & ImGuiConfigFlags_ViewportsEnable) {
    ImGui::UpdatePlatformWindows();
    ImGui::RenderPlatformWindowsDefault();
  }

  // Remove deleted textures from clients
  for (uint32_t i(0); i < RemoteClient::Client::GetCountMax(); ++i) {
    RemoteClient::Client& client = RemoteClient::Client::Get(i);
    client.mTextureTable.clear();
  }

  NetImguiServer::Network::Shutdown();
  NetImguiServer::UI::Shutdown();
  NetImguiServer::Config::Client::Clear();
  RemoteClient::Client::Shutdown();
  HAL_Shutdown();
}

//=================================================================================================
// INIT CLIENT CONFIG FROM STRING
// Take a commandline string, and create a ClientConfig from it.
// Simple format of (Hostname);(HostPort)
bool AddTransientClientConfigFromString(const char* string)
//=================================================================================================
{
  NetImguiServer::Config::Client cmdlineClient;
  const char* zEntryStart = string;
  const char* zEntryCur = string;
  int paramIndex = 0;
  cmdlineClient.mConfigType =
      NetImguiServer::Config::Client::eConfigType::Transient;
  NetImgui::Internal::StringCopy(cmdlineClient.mClientName, "Commandline");

  while (*zEntryCur != 0) {
    zEntryCur++;
    // Skip commandline preamble holding path to executable
    if (*zEntryCur == ' ' && *(zEntryCur + 1) != 0) {
      zEntryStart = zEntryCur + 1;
    }
    if ((*zEntryCur == ';' || *zEntryCur == 0)) {
      if (paramIndex == 0)
        NetImgui::Internal::StringCopy(cmdlineClient.mHostName, zEntryStart,
                                       zEntryCur - zEntryStart);
      else if (paramIndex == 1)
        cmdlineClient.mHostPort = static_cast<uint32_t>(atoi(zEntryStart));

      cmdlineClient.mConnectAuto =
          paramIndex >=
          1;  // Mark valid for connexion as soon as we have a HostAddress
      zEntryStart = zEntryCur + 1;
      paramIndex++;
    }
  }

  if (cmdlineClient.mConnectAuto) {
    NetImguiServer::Config::Client::SetConfig(cmdlineClient);
    return true;
  }

  return false;
}

//=================================================================================================
// DRAW CLIENT BACKGROUND
// Create a separate Dear ImGui drawing context, to generate a commandlist that
// will fill the RenderTarget with a specific background picture
void DrawClientBackground(RemoteClient::Client& client)
//=================================================================================================
{
  NetImgui::Internal::CmdBackground* pBGUpdate =
      client.mPendingBackgroundIn.Release();
  if (pBGUpdate) {
    client.mBGSettings = *pBGUpdate;
    client.mBGNeedUpdate = true;
    netImguiDeleteSafe(pBGUpdate);
  }

  if (client.mpBGContext == nullptr) {
    client.mpBGContext = ImGui::CreateContext(ImGui::GetIO().Fonts);
    client.mpBGContext->IO.DeltaTime = 1 / 30.f;
    client.mpBGContext->IO.IniFilename =
        nullptr;  // Disable server ImGui ini settings (not really needed, and
                  // avoid imgui.ini filename conflicts)
    client.mpBGContext->IO.LogFilename = nullptr;
    client.mpBGContext->IO.BackendFlags |=
        ImGuiBackendFlags_RendererHasTextures;
  }

  // Detect when the main font textures was updated,
  // in which case we need to re-generate the BG DrawData
  const ImVector<ImTextureData*>& MainTextures =
      ImGui::GetPlatformIO().Textures;
  NetImgui::Internal::ScopedImguiContext scopedCtx(client.mpBGContext);
  const ImVector<ImTextureData*>& BGTextures = ImGui::GetPlatformIO().Textures;
  for (ImTextureData* texData : BGTextures) {
    client.mBGNeedUpdate |= MainTextures.contains(texData) == false;
  }

  // Update the BG DrawData
  // Note: Maybe we should try rendering it directly in client windows every
  // frame instead, to keep things simpler
  if (client.mBGNeedUpdate) {
    ImGui::GetIO().DisplaySize = ImVec2(client.mAreaSizeX, client.mAreaSizeY);
    ImGui::NewFrame();
    ImGui::SetNextWindowPos(ImVec2(0, 0));
    ImGui::SetNextWindowSize(ImVec2(client.mAreaSizeX, client.mAreaSizeY));
    ImGui::Begin("Background", nullptr,
                 ImGuiWindowFlags_NoDecoration | ImGuiWindowFlags_NoInputs |
                     ImGuiWindowFlags_NoNav | ImGuiWindowFlags_NoBackground |
                     ImGuiWindowFlags_NoSavedSettings);
    // Look for the desired texture (and use default if not found)
    auto texIt = client.mTextureTable.find(client.mBGSettings.mTextureId);
    const ServerTexture* pTexture = texIt == client.mTextureTable.end()
                                        ? UI::GetBackgroundTexture()
                                        : texIt->second;
    UI::DrawCenteredBackground(pTexture,
                               ImVec4(client.mBGSettings.mTextureTint[0],
                                      client.mBGSettings.mTextureTint[1],
                                      client.mBGSettings.mTextureTint[2],
                                      client.mBGSettings.mTextureTint[3]));
    ImGui::End();
    ImGui::Render();
    client.mBGNeedUpdate = false;
  }
}

//=================================================================================================
void UpdateServerTextures()
//=================================================================================================
{
  for (int i(gServerTextures.size() - 1); i >= 0; --i) {
    if (gServerTextures[i]) {
      ServerTexture& ServerTex = *gServerTextures[i];

      // Release un-needed pixel data once it has been processed by backend
      if (ServerTex.mIsUpdatable == false &&
          ServerTex.mTexData.Pixels != nullptr &&
          ServerTex.mTexData.Status == ImTextureStatus_OK) {
        ServerTex.mTexData.DestroyPixels();
      }

      // Backend deleted the texture, remove it from our list
      else if (ServerTex.mTexData.Status == ImTextureStatus_Destroyed) {
        ImGui::UnregisterUserTexture(&ServerTex.mTexData);
        delete gServerTextures[i];
        gServerTextures[i] = nullptr;
      }

      // Send deletion request to backend
      else if (ServerTex.mTexData.WantDestroyNextFrame) {
        const RemoteClient::Client* Client =
            ServerTex.mOwnerClientIndex >= 0
                ? &RemoteClient::Client::Get(ServerTex.mOwnerClientIndex)
                : nullptr;
        if (!Client || !Client->mpImguiDrawData ||
            Client->mpImguiDrawData->mFrameIndex > ServerTex.mLastFrameUsed) {
          if (ServerTex.mTexData.UnusedFrames++ > 0) {
            ServerTex.mTexData.Status = ImTextureStatus_WantDestroy;
          }
        }
      }
    }

    // Remove release textures (null) from our list
    if (gServerTextures[i] == nullptr) {
      ImSwap(gServerTextures[i], gServerTextures[gServerTextures.Size - 1]);
      gServerTextures.pop_back();
    }
  }
}

//=================================================================================================
// Default texture creation behavior, relying on Dear ImGui backend to do
// the heavy lifting of texture creation and management
bool CreateTexture_Default(ServerTexture& serverTexture,
                           const NetImgui::Internal::CmdTexture& cmdTexture,
                           uint32_t customDataSize)
//=================================================================================================
{
  if (!serverTexture.mIsCustom && cmdTexture.mpTextureData.Get() != nullptr) {
    serverTexture.mIsUpdatable = cmdTexture.mUpdatable;
    serverTexture.mTexData.Create(ImTextureFormat::ImTextureFormat_RGBA32,
                                  cmdTexture.mWidth, cmdTexture.mHeight);
    if (cmdTexture.mFormat == ImTextureFormat::ImTextureFormat_RGBA32) {
      memcpy(serverTexture.mTexData.Pixels, cmdTexture.mpTextureData.Get(),
             serverTexture.mTexData.GetSizeInBytes());
    } else if (cmdTexture.mFormat == ImTextureFormat::ImTextureFormat_Alpha8) {
      const uint8_t* pSrcCur = cmdTexture.mpTextureData.Get();
      uint32_t* pDestCur =
          reinterpret_cast<uint32_t*>(serverTexture.mTexData.GetPixels());
      uint32_t* pDestEnd = &pDestCur[cmdTexture.mHeight * cmdTexture.mWidth];
      while (pDestCur < pDestEnd) {
        *pDestCur++ = 0x00FFFFFF | (uint64_t(*pSrcCur++) << 24);
      }
    } else {
      IM_ASSERT_USER_ERROR(0, "Unsupported format");
    }

    serverTexture.mTexData.Status = ImTextureStatus_WantCreate;
    serverTexture.mTexData.UseColors =
        cmdTexture.mFormat == ImTextureFormat::ImTextureFormat_RGBA32;
    ImGui::RegisterUserTexture(&serverTexture.mTexData);
    return true;
  }
  return false;
}

//=================================================================================================
ServerTexture* CreateTexture(const NetImgui::Internal::CmdTexture& cmdTexture,
                             uint32_t textureDataSize)
//=================================================================================================
{
  ServerTexture* serverTex(new ServerTexture);
  if (serverTex) {
    serverTex->mIsCustom =
        cmdTexture.mFormat == NetImgui::eTexFormat::kTexFmtCustom;
    serverTex->mClientTexID = cmdTexture.mTextureClientID;
    serverTex->mIsUpdatable = false;
    if (CreateTexture_Custom(*serverTex, cmdTexture, textureDataSize) ||
        CreateTexture_Default(*serverTex, cmdTexture, textureDataSize)) {
      gServerTextures.push_back(serverTex);
    } else {
      delete serverTex;
      serverTex = NULL;
    }
  }
  return serverTex;
}

//=================================================================================================
// Initialize all needed fonts by the NetImguiServer application
void LoadFonts()
//=================================================================================================
{
  ImFontConfig fontConfig;
  ImFontAtlas* pFontAtlas = ImGui::GetIO().Fonts;

  // Add Fonts here...
  // Using a different default font (provided with Dear ImGui)
  NetImgui::Internal::StringCopy(fontConfig.Name, "Roboto Medium");
  if (!pFontAtlas->AddFontFromMemoryCompressedTTF(Roboto_Medium_compressed_data,
                                                  Roboto_Medium_compressed_size,
                                                  0.f, &fontConfig)) {
    pFontAtlas->AddFontDefault(&fontConfig);
  }
}

//=================================================================================================
void UpdateWindowPlacement(int x, int y, int w, int h, bool isMaximized)
//=================================================================================================
{
  NetImguiServer::Config::Server::sWindowPlacement[0] = x;
  NetImguiServer::Config::Server::sWindowPlacement[1] = y;
  NetImguiServer::Config::Server::sWindowPlacement[2] = w;
  NetImguiServer::Config::Server::sWindowPlacement[3] = h;
  NetImguiServer::Config::Server::sWindowMaximized = isMaximized;
  NetImguiServer::Config::Client::SaveAll();
}

//=================================================================================================
WindowPlacement GetWindowPlacement()
//=================================================================================================
{
  if (!gLoadedConfigOnce) {
    NetImguiServer::Config::Client::LoadAll();
    gLoadedConfigOnce = true;
  }

  WindowPlacement wp;
  wp.x = NetImguiServer::Config::Server::sWindowPlacement[0];
  wp.y = NetImguiServer::Config::Server::sWindowPlacement[1];
  wp.w = ImMax(100, NetImguiServer::Config::Server::sWindowPlacement[2]);
  wp.h = ImMax(100, NetImguiServer::Config::Server::sWindowPlacement[3]);
  wp.isMaximized = NetImguiServer::Config::Server::sWindowMaximized;
  return wp;
}
//=================================================================================================
// UPDATE REMOTE CONTENT
// Create a render target for each connected remote client once, and update it
// every frame with the last drawing commands received from it. This Render
// Target will then be used normally as the background image of each client
// window renderered by this Server
void UpdateClientDraw()
//=================================================================================================
{
  for (uint32_t i(0); i < RemoteClient::Client::GetCountMax(); ++i) {
    RemoteClient::Client& client = RemoteClient::Client::Get(i);
    if (client.mbIsConnected) {
      if (client.mbIsReleased) {
        client.Uninitialize();
      } else {
        client.ProcessPendingTextureCmds();
        if (client.mbIsVisible) {
          // Update the RenderTarget destination of each client, of size was
          // updated
          if (client.mAreaSizeX > 0 && client.mAreaSizeY > 0 &&
              (!client.mpHAL_AreaRT ||
               client.mAreaRTSizeX != client.mAreaSizeX ||
               client.mAreaRTSizeY != client.mAreaSizeY)) {
            if (HAL_CreateRenderTarget(client.mAreaSizeX, client.mAreaSizeY,
                                       client.mpHAL_AreaRT,
                                       client.mHAL_AreaTexture)) {
              client.mAreaRTSizeX = client.mAreaSizeX;
              client.mAreaRTSizeY = client.mAreaSizeY;
              client.mLastUpdateTime =
                  std::chrono::steady_clock::now() -
                  std::chrono::hours(1);  // Will redraw the client
              client.mBGNeedUpdate = true;
            }
          }

          // Render the remote results
          ImDrawData* pDrawData =
              client.GetImguiDrawData(gServerTextureEmpty->mTexData.GetTexID());
          if (pDrawData) {
            DrawClientBackground(client);
            HAL_RenderDrawData(client, pDrawData);
          }
        }
      }
    }
  }

  UpdateServerTextures();
}
}  // namespace App
}  // namespace NetImguiServer
