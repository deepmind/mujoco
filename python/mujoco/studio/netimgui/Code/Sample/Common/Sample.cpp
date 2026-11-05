//=================================================================================================
// SAMPLE
//-------------------------------------------------------------------------------------------------
// Common code shared by all samples
//=================================================================================================

#include "Sample.h"

#include <NetImgui_Api.h>
#include <math.h>

#include "../../ServerApp/Source/Fonts/Roboto_Medium.cpp"

namespace Sample {
//=================================================================================================
// Constructor
//-------------------------------------------------------------------------------------------------
//
//=================================================================================================
Base::Base(const char* sampleName) : mSampleName(sampleName) {
#if NETIMGUI_ENABLED
  mConnect_PortClient = NetImgui::kDefaultClientPort;
  mConnect_PortServer = NetImgui::kDefaultServerPort;
#endif
}

//=================================================================================================
// Startup
//-------------------------------------------------------------------------------------------------
//
//=================================================================================================
bool Base::Startup() {
#if NETIMGUI_ENABLED
  if (!NetImgui::Startup()) return false;
#endif
  AddFont();
  return true;
}

//=================================================================================================
// Shutdown
//-------------------------------------------------------------------------------------------------
//
//=================================================================================================
void Base::Shutdown() {
#if NETIMGUI_ENABLED
  NetImgui::Shutdown();
#endif
}

//=================================================================================================
// AddFont
//-------------------------------------------------------------------------------------------------
// Add and configure the fonts wanted in the demo.
// Method can be overriden to use different font, byt default to Roboto 18pts
//
// Note:	Since Dear ImGui 1.92+, we do not need to manage font
// scaling/dpi at all,
//			It is automatically done, and NetImgui Server also
//assign desired DPI behind the scene.
//=================================================================================================
void Base::AddFont() {
  constexpr float kFontPixelSize = 16.f;
  ImFontConfig FontConfig = {};
// Using Roboto Font for prettier results
#if 1
  // Note: Using memcpy to avoid warnings related to OS string copy variations,
  //		and cannot rely on 'NetImgui::Internal::StringCopy' that can
  //handle this problem 		because 'SampleDisabled' couln't compile properly
  const char FontName[] = "Roboto Medium";
  memcpy(FontConfig.Name, FontName, sizeof(FontName));
  ImGui::GetIO().Fonts->AddFontFromMemoryCompressedTTF(
      Roboto_Medium_compressed_data, Roboto_Medium_compressed_size,
      kFontPixelSize, &FontConfig);

// But can as easily rely on the default font
#else
  FontConfig.SizePixels = kFontPixelSize;
  FontAtlas->AddFontDefault(&FontConfig);
#endif
}

//=================================================================================================
// Draw_Connect
//-------------------------------------------------------------------------------------------------
// Function called by all samples, to display the Connection Options, and some
// other default MainMenu entries.
//=================================================================================================
void Base::Draw_Connect() {
#if NETIMGUI_ENABLED
  ImGui::PushStyleVar(ImGuiStyleVar_FramePadding, ImVec2(3, 6));
  if (ImGui::BeginMainMenuBar()) {
    ImGui::AlignTextToFramePadding();
    ImGui::TextColored(ImVec4(0.1, 1, 0.1, 1), "%s", mSampleName);
    ImGui::SameLine(0, 32);

    //-----------------------------------------------------------------------------------------
    if (NetImgui::IsConnected())
    //-----------------------------------------------------------------------------------------
    {
      ImGui::TextUnformatted("Status: Connected");
      ImGui::PushStyleVar(ImGuiStyleVar_FramePadding, ImVec2(3, 3));
      ImGui::SetCursorPosY(3);
      if (ImGui::Button(" Disconnect ")) {
        NetImgui::Disconnect();
      }
      ImGui::PopStyleVar();
    }

    //-----------------------------------------------------------------------------------------
    else if (NetImgui::IsConnectionPending())
    //-----------------------------------------------------------------------------------------
    {
      ImGui::TextUnformatted("Status: Waiting Server");
      ImGui::PushStyleVar(ImGuiStyleVar_FramePadding, ImVec2(3, 3));
      ImGui::SetCursorPosY(3);
      if (ImGui::Button(" Cancel ")) {
        NetImgui::Disconnect();
      }
      ImGui::PopStyleVar();
    }

    //-----------------------------------------------------------------------------------------
    else  // No connection
    //-----------------------------------------------------------------------------------------
    {
      //-------------------------------------------------------------------------------------
      if (ImGui::BeginMenu("[ Connect To ]"))
      //-------------------------------------------------------------------------------------
      {
        ImGui::TextColored(ImVec4(0.1, 1, 0.1, 1), "Server Settings");
        ImGui::InputText("Hostname", mConnect_HostnameServer,
                         sizeof(mConnect_HostnameServer));
        if (ImGui::IsItemHovered())
          ImGui::SetTooltip(
              "Address of PC running the netImgui server application. Can be "
              "an IP like 127.0.0.1");
        ImGui::InputInt("Port", &mConnect_PortServer);
        ImGui::NewLine();
        ImGui::Separator();
        if (ImGui::Button("Connect",
                          ImVec2(ImGui::GetContentRegionAvail().x, 0))) {
          NetImgui::ConnectToApp(mSampleName, mConnect_HostnameServer,
                                 mConnect_PortServer, mCallback_ThreadLaunch);
        }
        ImGui::EndMenu();
      }

      if (ImGui::IsItemHovered())
        ImGui::SetTooltip(
            "Attempt a connection to a remote netImgui server at the provided "
            "address.");

      //-------------------------------------------------------------------------------------
      if (ImGui::BeginMenu("[  Wait For  ]"))
      //-------------------------------------------------------------------------------------
      {
        ImGui::TextColored(ImVec4(0.1, 1, 0.1, 1), "Client Settings");
        ImGui::InputInt("Port", &mConnect_PortClient);
        ImGui::NewLine();
        ImGui::Separator();
        if (ImGui::Button("Listen",
                          ImVec2(ImGui::GetContentRegionAvail().x, 0))) {
          NetImgui::ConnectFromApp(mSampleName, mConnect_PortClient,
                                   mCallback_ThreadLaunch);
        }
        ImGui::EndMenu();
      }
      if (ImGui::IsItemHovered())
        ImGui::SetTooltip(
            "Start listening for a connection request by a remote netImgui "
            "server, on the provided Port.");
    }

    ImGui::SameLine(0, 40);
    ImGui::PushStyleColor(ImGuiCol_Border, ImVec4(0.8, 0.8, 0.8, 0.9));
    ImGui::PushStyleVar(ImGuiStyleVar_FramePadding, ImVec2(3, 3));
    ImGui::PushStyleVar(ImGuiStyleVar_FrameBorderSize,
                        mbShowDemoWindow ? 1.f : 0.f);
    ImGui::SetCursorPosY(3);
    if (ImGui::Button(" Show ImGui Demo ")) {
      mbShowDemoWindow = !mbShowDemoWindow;
    }
    ImGui::PopStyleColor();
    ImGui::PopStyleVar(2);
    ImGui::EndMainMenuBar();
  }
  ImGui::PopStyleVar();
#endif  // #if NETIMGUI_ENABLED
  if (mbShowDemoWindow) {
    ImGui::ShowDemoWindow(&mbShowDemoWindow);
  }
}

};  // namespace Sample