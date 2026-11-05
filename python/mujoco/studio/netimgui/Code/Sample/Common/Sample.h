#pragma once

#include <NetImgui_Api.h>

// Reusing internal NetImgui functions for TextureID conversion and String copy.
// Normally, you wouldn't include this file
#include <Private/NetImgui_Shared.h>

// Forward declares when NetImgui is not enabled
// When NetImgui is disabled, it doesn't include these needed headers
#if !NETIMGUI_ENABLED
#include "imgui.h"
#endif

namespace Sample {

class Base {
 public:
  Base(const char* sampleName);  //!< Constructor receiving pointer to constant
                                 //!< string that must remains valid
  virtual bool Startup();        //!< Called once when starting
  virtual void Shutdown();       //!< Called once when exiting
  virtual void Draw() = 0;       //!< Each sample should have their Dear ImGui
                                 //!< drawing routines in this overloaded method
  virtual void AddFont();        //!< Called on startup to add the wanted font

 protected:
  void Draw_Connect();  //!< Display UI for initiating a connection to the
                        //!< remote NetImgui server application
  const char* mSampleName =
      nullptr;  //!< Name displayed in the Main Menu bar (must receive string
                //!< pointer in constructor that remains valid)
  bool mbShowDemoWindow =
      !NETIMGUI_ENABLED;  //!< If we should show the Dear ImGui demo window

#if NETIMGUI_ENABLED
  NetImgui::ThreadFunctPtr mCallback_ThreadLaunch =
      nullptr;  //!< [Optional] Thread launcher callback assigned on NetImgui
                //!< connection. Used to start a new thread for coms with
                //!< NetImgui server
  char mConnect_HostnameServer[128] = {
      "localhost"};  //!< IP/Hostname used to send a connection request when
                     //!< when trying to reach the server
  int mConnect_PortServer = 0;  //!< Port used to send a connection request when
                                //!< when trying to reach the server
  int mConnect_PortClient =
      0;  //!< Port opened when waiting for a server connection request
#endif
};
};  // namespace Sample

Sample::Base& GetSample();  // Each Sample must implement this function and
                            // return a valid sample object

#include <Client/Private/NetImgui_WarningDisable.h>
