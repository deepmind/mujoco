// Google modifications:
//  - Updated nlohmann/json include path for Google third_party layout.
#include "NetImguiServer_Config.h"

#include <NetImgui_Api.h>

#include <fstream>
#include <mutex>
#include <nlohmann/json.hpp>

#include "NetImguiServer_RemoteClient.h"

namespace NetImguiServer {
namespace Config {

static ImVector<Client*> gConfigList;
static std::mutex gConfigLock;
static Client::RuntimeID gRuntimeID = static_cast<Client::RuntimeID>(1);

static constexpr char kConfigField_ServerPort[] = "ServerPort";
static constexpr char kConfigField_ServerRefreshActive[] = "RefreshFPSActive";
static constexpr char kConfigField_ServerRefreshInactive[] =
    "RefreshFPSInactive";
static constexpr char kConfigField_ServerDPIScaleRatio[] = "DPIScaleRatio";
static constexpr char kConfigField_ServerCompressionEnable[] =
    "CompressionEnable";
static constexpr char kConfigField_ServerFontSize[] = "ServerFontSize";
static constexpr char kConfigField_ServerWindowPlacementX[] =
    "ServerWindowPlacementX";
static constexpr char kConfigField_ServerWindowPlacementY[] =
    "ServerWindowPlacementY";
static constexpr char kConfigField_ServerWindowPlacementW[] =
    "ServerWindowPlacementW";
static constexpr char kConfigField_ServerWindowPlacementH[] =
    "ServerWindowPlacementH";
static constexpr char kConfigField_ServerWindowMaximized[] =
    "ServerWindowMaximized";

static constexpr char kConfigField_Note[] = "Note";
static constexpr char kConfigField_Version[] = "Version";
static constexpr char kConfigField_Configs[] = "Configs";
static constexpr char kConfigField_Name[] = "Name";
static constexpr char kConfigField_Hostname[] = "Hostname";
static constexpr char kConfigField_Hostport[] = "HostPort";
static constexpr char kConfigField_AutoConnect[] = "Auto";
static constexpr char kConfigField_BlockTakeover[] = "BlockTakeover";
static constexpr char kConfigField_DPIScaleEnabled[] = "DPIScaleEnabled";

uint32_t Server::sPort = NetImgui::kDefaultServerPort;
float Server::sRefreshFPSActive = 30.f;
float Server::sRefreshFPSInactive = 30.f;
float Server::sDPIScaleRatio = 1.f;
bool Server::sCompressionEnable = true;
float Server::sFontSize = 16.f;
int Server::sWindowPlacement[4] = {100, 100, 1280, 1024};
bool Server::sWindowMaximized = false;

//=================================================================================================
// The user config is created when the main config file is readonly
// Allows having a distributed config file that cannot be touched by users
static const char* GetConfigFilename(Client::eConfigType configFileType)
//=================================================================================================
{
  if (configFileType == Client::eConfigType::Local) {
    return "netImgui.cfg";
  } else if (configFileType == Client::eConfigType::Local2nd) {
    return "netImgui_2.cfg";
  } else {
    static char sUserSettingFile[1024];
    const char* userSettingFolder =
        NetImguiServer::App::HAL_GetUserSettingFolder();
    NetImgui::Internal::StringFormat(
        sUserSettingFile,
        (userSettingFolder ? "%s\\netImgui.cfg" : "netImgui_2.cfg"),
        userSettingFolder);
    return sUserSettingFile;
  }
}

//=================================================================================================
// Find entry index with same configId. (-1 if not found)
// Note: 'gConfigLock' should have already locked before calling this
static int FindClientIndex(uint32_t configID)
//=================================================================================================
{
  if (configID != Client::kInvalidRuntimeID) {
    for (int i(0); i < gConfigList.size(); ++i) {
      if (gConfigList[i] && gConfigList[i]->mRuntimeID == configID) return i;
    }
  }
  return -1;
}

//=================================================================================================
template <typename TType>
TType GetPropertyValue(const nlohmann::json& config, const char* zPropertyName,
                       const TType& valueDefault)
//=================================================================================================
{
  const auto& valueNode = config.find(zPropertyName);
  return valueNode != config.end() ? valueNode->get<TType>() : valueDefault;
}

//=================================================================================================
Client::Client()
    //=================================================================================================
    : mHostPort(NetImgui::kDefaultClientPort),
      mRuntimeID(kInvalidRuntimeID),
      mConfigType(NetImguiServer::Config::Client::eConfigType::Pending),
      mDPIScaleEnabled(true),
      mBlockTakeover(false),
      mReadOnly(false),
      mConnectAuto(false),
      mConnectRequest(false),
      mConnectForce(false),
      mConnectStatus(eStatus::Disconnected) {
  NetImgui::Internal::StringCopy(mClientName, "New Client");
  NetImgui::Internal::StringCopy(mHostName, "localhost");
}

//=================================================================================================
void Client::SetConfig(const Client& config)
//=================================================================================================
{
  std::lock_guard<std::mutex> guard(gConfigLock);

  // Only allow 1 transient connection to keep things clean
  for (int i = 0; config.IsTransient() && i < gConfigList.size(); ++i) {
    if (gConfigList[i] && gConfigList[i]->IsTransient()) {
      NetImgui::Internal::netImguiDelete(gConfigList[i]);
      gConfigList.erase(&gConfigList[i]);
    }
  }

  int index = FindClientIndex(config.mRuntimeID);
  // Config not found, add it to our list
  if (index == -1) {
    index = gConfigList.size();
    gConfigList.push_back(
        NetImgui::Internal::netImguiNew<NetImguiServer::Config::Client>());
  }

  // Update the entry
  *gConfigList[index] = config;
  gConfigList[index]->mRuntimeID =
      (config.mRuntimeID == kInvalidRuntimeID ? gRuntimeID++
                                              : config.mRuntimeID);
}

//=================================================================================================
void Client::DelConfig(uint32_t configID)
//=================================================================================================
{
  std::lock_guard<std::mutex> guard(gConfigLock);
  int index = FindClientIndex(configID);
  if (index != -1) {
    NetImgui::Internal::netImguiDelete(gConfigList[index]);
    gConfigList.erase(&gConfigList[index]);
  }
}

//=================================================================================================
bool Client::GetConfigByID(uint32_t configID, Client& config)
//=================================================================================================
{
  std::lock_guard<std::mutex> guard(gConfigLock);
  int index = FindClientIndex(configID);
  if (index != -1) {
    config = *gConfigList[index];
    return true;
  }
  return false;
}

//=================================================================================================
bool Client::GetConfigByIndex(uint32_t index, Client& config)
//=================================================================================================
{
  std::lock_guard<std::mutex> guard(gConfigLock);
  if (index < static_cast<uint32_t>(gConfigList.size())) {
    config = *gConfigList[index];
    return true;
  }
  return false;
}

//=================================================================================================
uint32_t Client::GetConfigCount()
//=================================================================================================
{
  std::lock_guard<std::mutex> guard(gConfigLock);
  return gConfigList.size();
}

//=================================================================================================
bool Client::GetProperty_BlockTakeover(uint32_t configID)
//=================================================================================================
{
  std::lock_guard<std::mutex> guard(gConfigLock);
  int index = FindClientIndex(configID);
  if (index != -1) {
    return gConfigList[index]->mBlockTakeover;
  }
  return false;
}

//=================================================================================================
void Client::SetProperty_Status(uint32_t configID, eStatus Status)
//=================================================================================================
{
  std::lock_guard<std::mutex> guard(gConfigLock);
  int index = FindClientIndex(configID);
  if (index != -1) {
    gConfigList[index]->mConnectStatus = Status;
  }
}

//=================================================================================================
void Client::SetProperty_ConnectAuto(uint32_t configID, bool value)
//=================================================================================================
{
  std::lock_guard<std::mutex> guard(gConfigLock);
  int index = FindClientIndex(configID);
  if (index != -1) {
    gConfigList[index]->mConnectAuto = value;
  }
}

//=================================================================================================
void Client::SetProperty_ConnectRequest(uint32_t configID, bool value,
                                        bool force)
//=================================================================================================
{
  std::lock_guard<std::mutex> guard(gConfigLock);
  int index = FindClientIndex(configID);
  if (index != -1) {
    gConfigList[index]->mConnectRequest = value && !force;
    gConfigList[index]->mConnectForce = value && force;
    gConfigList[index]->mConnectLastTime = std::chrono::steady_clock::now();
  }
}

//=================================================================================================
bool Client::ShouldSave(eConfigType fileConfigType) const
//=================================================================================================
{
  return mConfigType == fileConfigType || mConfigType == eConfigType::Pending;
}

//=================================================================================================
void Client::SaveAll()
//=================================================================================================
{
  std::lock_guard<std::mutex> guard(gConfigLock);
  std::ofstream localFile(GetConfigFilename(eConfigType::Local));
  bool localIsWritable = localFile.is_open();
  localFile.close();

  std::ifstream local2ndFileExist(GetConfigFilename(eConfigType::Local2nd));
  bool local2ndExist = local2ndFileExist.is_open();
  bool local2ndIsWritable = false;
  local2ndFileExist.close();

  // Try saving into default config file
  SaveConfigFile(eConfigType::Local, localIsWritable);
  // And then in 2nd workingdir user file when 1st one is read only
  if (!localIsWritable || local2ndExist) {
    std::ofstream local2ndFile(GetConfigFilename(eConfigType::Local2nd));
    SaveConfigFile(eConfigType::Local2nd,
                   !localIsWritable && local2ndIsWritable);
  }

  // Finally saved the shared config into user folder
  SaveConfigFile(eConfigType::Shared, !localIsWritable && !local2ndIsWritable);
}

//=================================================================================================
void Client::SaveConfigFile(eConfigType configFileType,
                            bool writeServerSettings)
//=================================================================================================
{
  nlohmann::json configRoot;
  configRoot[kConfigField_Version] = eVersion::_Latest;
  configRoot[kConfigField_Note] =
      configFileType == eConfigType::Local
          ? "NetImgui Server's list of Clients (Using JSON format) Local File."
      : configFileType == eConfigType::Local
          ? "NetImgui Server's list of Clients (Using JSON format) 2nd Local "
            "File."
          : "NetImgui Server's list of Clients (Using JSON format) Shared "
            "File.";
  if (writeServerSettings) {
    configRoot[kConfigField_ServerPort] = Server::sPort;
    configRoot[kConfigField_ServerRefreshActive] = Server::sRefreshFPSActive;
    configRoot[kConfigField_ServerRefreshInactive] =
        Server::sRefreshFPSInactive;
    configRoot[kConfigField_ServerDPIScaleRatio] = Server::sDPIScaleRatio;
    configRoot[kConfigField_ServerCompressionEnable] =
        Server::sCompressionEnable;
    configRoot[kConfigField_ServerFontSize] = Server::sFontSize;
    configRoot[kConfigField_ServerWindowPlacementX] =
        Server::sWindowPlacement[0];
    configRoot[kConfigField_ServerWindowPlacementY] =
        Server::sWindowPlacement[1];
    configRoot[kConfigField_ServerWindowPlacementW] =
        Server::sWindowPlacement[2];
    configRoot[kConfigField_ServerWindowPlacementH] =
        Server::sWindowPlacement[3];
    configRoot[kConfigField_ServerWindowMaximized] = Server::sWindowMaximized;
  }

  int clientToSaveCount(0);
  for (int i(0); i < gConfigList.size(); ++i) {
    Client* pConfig = gConfigList[i];
    if (pConfig && pConfig->ShouldSave(configFileType)) {
      auto& config = configRoot[kConfigField_Configs][clientToSaveCount++] =
          nullptr;
      config[kConfigField_Name] = pConfig->mClientName;
      config[kConfigField_Hostname] = pConfig->mHostName;
      config[kConfigField_Hostport] = pConfig->mHostPort;
      config[kConfigField_AutoConnect] = pConfig->mConnectAuto;
      config[kConfigField_BlockTakeover] = pConfig->mBlockTakeover;
      config[kConfigField_DPIScaleEnabled] = pConfig->mDPIScaleEnabled;
    }
  }

  std::ofstream outputFile(GetConfigFilename(configFileType));
  if (outputFile.is_open()) {
    outputFile << configRoot.dump(4);
    for (int i(0); clientToSaveCount > 0 && i < gConfigList.size(); ++i) {
      Client* pConfig = gConfigList[i];
      if (pConfig && pConfig->ShouldSave(configFileType)) {
        pConfig->mReadOnly = false;
        pConfig->mConfigType = configFileType;
      }
    }
  }
}

//=================================================================================================
void Client::LoadAll()
//=================================================================================================
{
  Clear();
  std::lock_guard<std::mutex> guard(gConfigLock);
  LoadConfigFile(eConfigType::Local);
  LoadConfigFile(eConfigType::Local2nd);
  LoadConfigFile(eConfigType::Shared);
}

//=================================================================================================
void Client::LoadConfigFile(eConfigType configFileType)
//=================================================================================================
{
  nlohmann::json configRoot;
  const char* filename = GetConfigFilename(configFileType);
  std::ifstream inputFile(filename);
  if (!inputFile.is_open() || inputFile.eof()) return;

  configRoot = nlohmann::json::parse(inputFile, nullptr, false);
  inputFile.close();

  std::ofstream outputFile(filename, std::ios_base::app);
  bool isWritable = outputFile.is_open() && !outputFile.eof();
  outputFile.close();

  uint32_t configVersion =
      GetPropertyValue(configRoot, kConfigField_Version, 0u);
  Server::sPort = GetPropertyValue(configRoot, kConfigField_ServerPort,
                                   NetImgui::kDefaultServerPort);
  Server::sRefreshFPSActive = GetPropertyValue(
      configRoot, kConfigField_ServerRefreshActive, Server::sRefreshFPSActive);
  Server::sRefreshFPSInactive =
      GetPropertyValue(configRoot, kConfigField_ServerRefreshInactive,
                       Server::sRefreshFPSInactive);
  Server::sDPIScaleRatio = GetPropertyValue(
      configRoot, kConfigField_ServerDPIScaleRatio, Server::sDPIScaleRatio);
  Server::sCompressionEnable =
      GetPropertyValue(configRoot, kConfigField_ServerCompressionEnable,
                       Server::sCompressionEnable);
  Server::sFontSize = GetPropertyValue(configRoot, kConfigField_ServerFontSize,
                                       Server::sFontSize);
  Server::sWindowPlacement[0] =
      GetPropertyValue(configRoot, kConfigField_ServerWindowPlacementX,
                       Server::sWindowPlacement[0]);
  Server::sWindowPlacement[1] =
      GetPropertyValue(configRoot, kConfigField_ServerWindowPlacementY,
                       Server::sWindowPlacement[1]);
  Server::sWindowPlacement[2] =
      GetPropertyValue(configRoot, kConfigField_ServerWindowPlacementW,
                       Server::sWindowPlacement[2]);
  Server::sWindowPlacement[3] =
      GetPropertyValue(configRoot, kConfigField_ServerWindowPlacementH,
                       Server::sWindowPlacement[3]);
  Server::sWindowMaximized = GetPropertyValue(
      configRoot, kConfigField_ServerWindowMaximized, Server::sWindowMaximized);

  if (configVersion >= static_cast<uint32_t>(eVersion::Initial)) {
    for (const auto& config : configRoot[kConfigField_Configs]) {
      gConfigList.push_back(
          NetImgui::Internal::netImguiNew<NetImguiServer::Config::Client>());
      Client* pConfig = gConfigList.back();
      pConfig->mRuntimeID = gRuntimeID++;

      if (config.find(kConfigField_Name) != config.end())
        NetImgui::Internal::StringCopy(
            pConfig->mClientName,
            config[kConfigField_Name].get<std::string>().c_str());

      if (config.find(kConfigField_Hostname) != config.end())
        NetImgui::Internal::StringCopy(
            pConfig->mHostName,
            config[kConfigField_Hostname].get<std::string>().c_str());

      pConfig->mHostPort =
          GetPropertyValue(config, kConfigField_Hostport, pConfig->mHostPort);
      pConfig->mConnectAuto = GetPropertyValue(config, kConfigField_AutoConnect,
                                               pConfig->mConnectAuto);
      pConfig->mDPIScaleEnabled = GetPropertyValue(
          config, kConfigField_DPIScaleEnabled, pConfig->mDPIScaleEnabled);
      pConfig->mBlockTakeover = GetPropertyValue(
          config, kConfigField_BlockTakeover, pConfig->mBlockTakeover);
      pConfig->mConfigType = configFileType;
      pConfig->mReadOnly = !isWritable;
    }
  }
}

//=================================================================================================
void Client::Clear()
//=================================================================================================
{
  std::lock_guard<std::mutex> guard(gConfigLock);
  while (gConfigList.size()) {
    NetImgui::Internal::netImguiDelete(gConfigList.back());
    gConfigList.pop_back();
  }
}

}  // namespace Config
}  // namespace NetImguiServer
