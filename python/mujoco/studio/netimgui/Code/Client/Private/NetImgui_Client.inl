
#include "NetImgui_Network.h"

namespace NetImgui { namespace Internal { namespace Client {

bool ClientInfo::IsConnected()const
{
	return mpSocketComs.load() != nullptr;
}

bool ClientInfo::IsConnectPending()const
{
	return mbComInitActive || mpSocketPending.load() != nullptr || mpSocketListen.load() != nullptr;
}

bool ClientInfo::IsActive()const
{
	return mbClientThreadActive || mbListenThreadActive;
}

bool ClientInfo::IsContextOverriden()const
{
	return mSavedContextValues.mSavedContext;
}

}}} // namespace NetImgui::Internal::Client
