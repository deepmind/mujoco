// Google modifications:
//  - Use ImGui::GetAllocatorFunctions() instead of ImGui::MemAlloc/MemFree in
//    netImguiNew/Delete to avoid data races on Dear ImGui global debug
//    allocation state from the background communication thread.
//  - OffsetPointer::ToPointer() validates that the data (offset received from
//    the network) is inside the command buffer.
//  - Fixed GetOff() mask truncating offsets to 32 bits.
#pragma once

#include <assert.h>
#include <string.h>

namespace NetImgui { namespace Internal
{

inline void* netImguiAlloc(size_t size)
{
	ImGuiMemAllocFunc allocFunc;
	ImGuiMemFreeFunc freeFunc;
	void* pUserData;
	ImGui::GetAllocatorFunctions(&allocFunc, &freeFunc, &pUserData);
	return allocFunc(size, pUserData);
}

inline void netImguiFree(void* pData)
{
	ImGuiMemAllocFunc allocFunc;
	ImGuiMemFreeFunc freeFunc;
	void* pUserData;
	ImGui::GetAllocatorFunctions(&allocFunc, &freeFunc, &pUserData);
	freeFunc(pData, pUserData);
}

template <typename TType, typename... Args> 
TType* netImguiNew(Args... args)
{
	return new( netImguiAlloc(sizeof(TType)) ) TType(args...);
}

template <typename TType> 
TType* netImguiSizedNew(size_t placementSize)
{
	return new( netImguiAlloc(placementSize > sizeof(TType) ? placementSize : sizeof(TType)) ) TType();
}

template <typename TType> 
void netImguiDelete(TType* pData)
{
	if( pData )
	{
		pData->~TType();
		netImguiFree(pData);
	}
}

template <typename TType> 
void netImguiDeleteSafe(TType*& pData)
{
	netImguiDelete(pData);
	pData = nullptr;
}

//=============================================================================
// Acquire ownership of the resource
//=============================================================================
template <typename TType>
TType* ExchangePtr<TType>::Release()
{
	return mpData.exchange(nullptr);
}

//-----------------------------------------------------------------------------
// Take ownership of the provided data.
// If there's a previous unclaimed pointer to some data, release it
//-----------------------------------------------------------------------------
template <typename TType>
void ExchangePtr<TType>::Assign(TType*& pNewData)
{
	netImguiDelete( mpData.exchange(pNewData) );
	pNewData = nullptr;
}

template <typename TType>
void ExchangePtr<TType>::Free()
{
	TType* pNull(nullptr);
	Assign(pNull);
}

template <typename TType>
ExchangePtr<TType>::~ExchangePtr()	
{ 
	Free();
}

//=============================================================================
//
//=============================================================================
template <typename TType>
OffsetPointer<TType>::OffsetPointer()
: mOffset(0)
{
	SetOff(0);
}

template <typename TType>
OffsetPointer<TType>::OffsetPointer(TType* pPointer)
{
	SetPtr(pPointer);
}

template <typename TType>
OffsetPointer<TType>::OffsetPointer(uint64_t offset)
{
	SetOff(offset);
}

template <typename TType>
void OffsetPointer<TType>::SetPtr(TType* pPointer)
{
	mOffset		= 0; // Remove 'offset flag' that can be left active on non 64bits pointer
	mPointer	= pPointer;
}

template <typename TType>
void OffsetPointer<TType>::SetComDataPtr(ComDataType* pPointer)
{
	SetPtr(reinterpret_cast<TType*>(pPointer));
}

template <typename TType>
void OffsetPointer<TType>::SetOff(uint64_t offset)
{
	mOffset = offset | 0x0000000000000001u;
}

template <typename TType>
uint64_t OffsetPointer<TType>::GetOff()const
{
	return mOffset & ~0x0000000000000001ull;
}

template <typename TType>
bool OffsetPointer<TType>::IsOffset()const
{
	return (mOffset & 0x0000000000000001u) != 0;
}

template <typename TType>
bool OffsetPointer<TType>::IsPointer()const
{
	return !IsOffset();
}

//=============================================================================
// Convert the offset to a pointer (or keep the pointer) of data inside the
// command buffer [pBufferStart, pBufferEnd). The offset comes from the network,
// so make sure the 'byteSize' bytes it points to are entirely in the buffer.
// Returns false (and sets a null pointer) when they are not.
//=============================================================================
template <typename TType>
bool OffsetPointer<TType>::ToPointer(const void* pBufferStart, const void* pBufferEnd, uint64_t byteSize)
{
	const uint64_t bufferStart	= reinterpret_cast<uint64_t>(pBufferStart);
	const uint64_t bufferEnd	= reinterpret_cast<uint64_t>(pBufferEnd);
	const uint64_t address		= IsOffset() ? reinterpret_cast<uint64_t>(&mPointer) + GetOff() : reinterpret_cast<uint64_t>(mPointer);
	const bool bValid			= address >= bufferStart && address <= bufferEnd && byteSize <= bufferEnd - address;
	SetPtr( bValid ? reinterpret_cast<TType*>(address) : nullptr );
	return bValid;
}

template <typename TType>
uint64_t OffsetPointer<TType>::ToOffset()
{
	assert(IsPointer());
	SetOff( reinterpret_cast<uint64_t>(mPointer) - reinterpret_cast<uint64_t>(&mPointer) );
	return mOffset;
}

template <typename TType>
TType* OffsetPointer<TType>::operator->()
{
	assert(IsPointer());
	return mPointer;
}

template <typename TType>
const TType* OffsetPointer<TType>::operator->()const
{
	assert(IsPointer());
	return mPointer;
}

template <typename TType>
TType* OffsetPointer<TType>::Get()
{
	assert(IsPointer());
	return mPointer;
}

template <typename TType>
const TType* OffsetPointer<TType>::Get()const
{
	assert(IsPointer());
	return mPointer;
}

template <typename TType>
const ComDataType* OffsetPointer<TType>::GetComData()const
{
	return reinterpret_cast<const ComDataType*>(Get());
}

template <typename TType>
TType& OffsetPointer<TType>::operator[](size_t index)
{
	assert(IsPointer());
	return mPointer[index];
}

template <typename TType>
const TType& OffsetPointer<TType>::operator[](size_t index)const
{
	assert(IsPointer());
	return mPointer[index];
}

//=============================================================================
template <typename TType, size_t TCount>
void Ringbuffer<TType,TCount>::AddData(const TType* pData, size_t& count)
//=============================================================================
{
	size_t i(0);
	while (i < count && (mPosLast - mPosCur < TCount)) {
		mBuffer[mPosLast % TCount] = pData[i];
		mPosLast++;
		i++;
	}
	count = i;
}

//=============================================================================
template <typename TType, size_t TCount>
bool Ringbuffer<TType,TCount>::ReadData(TType* pData)
//=============================================================================
{
	if (mPosCur < mPosLast) 
	{
		*pData = mBuffer[mPosCur % TCount];
		mPosCur++;
		return true;
	}
	return false;
}


//=============================================================================
// The _s string functions are a mess. There's really no way to do this right
// in a cross-platform way. Best solution I've found is to set just use
// strncpy, infer the buffer length, and null terminate. Still need to suppress
// the warning on Windows.
// See https://randomascii.wordpress.com/2013/04/03/stop-using-strncpy-already/
// and many other discussions online on the topic.
//=============================================================================
template <size_t charCount>
void StringCopy(char (&output)[charCount], const char* pSrc, size_t srcCharCount)
{
#if defined(__clang__)
	#pragma clang diagnostic push
	#pragma clang diagnostic ignored "-Wdeprecated-declarations"
#elif defined(_MSC_VER)
	#pragma warning (push)
	#pragma warning (disable: 4996)	// warning C4996: 'strncpy': This function or variable may be unsafe.
#endif

	size_t charToCopyCount = charCount < srcCharCount + 1 ? charCount : srcCharCount + 1;
	strncpy(output, pSrc, charToCopyCount - 1);
	output[charCount - 1] = 0;

#if defined(_MSC_VER) && defined(__clang__)
	#pragma clang diagnostic pop
#elif defined(_MSC_VER)
	#pragma warning (pop)
#endif
}

template <size_t charCount>
int StringFormat(char(&output)[charCount], char const* const format, ...)
{
#if defined(__clang__)
	#pragma clang diagnostic push
	#pragma clang diagnostic ignored "-Wformat-nonliteral"	
#endif

	va_list args;
    va_start(args, format);
	int w = vsnprintf(output, charCount, format, args);
	va_end(args);
	output[charCount - 1] = 0;
	return w;

#if defined(__clang__)
	#pragma clang diagnostic pop
#endif
}

//=============================================================================
//=============================================================================
template <typename IntType>
IntType DivUp(IntType Value, IntType Denominator)
{
	return (Value + Denominator - 1) / Denominator;
}

template <typename IntType>
IntType RoundUp(IntType Value, IntType Round)
{
	return DivUp(Value, Round) * Round;
}

union TextureCastHelperUnion
{
	const void*		TexData;
	ImTextureID 	TexID;
	ClientTextureID	TexClientID;
};

#if NETIMGUI_IMGUI_TEXTURES_ENABLED
NetImgui::eTexFormat ConvertTextureFormat(ImTextureFormat ImFormat)
{
	switch(ImFormat)
	{
		case ImTextureFormat_RGBA32: return eTexFormat::kTexFmtRGBA8;
		case ImTextureFormat_Alpha8: return eTexFormat::kTexFmtA8;
	}
	return eTexFormat::kTexFmtRGBA8;
}

ClientTextureID ConvertToClientTexID(const ImTextureRef& textureRef)
{
	static_assert(sizeof(uint64_t) >= sizeof(ImTextureID), "ImTextureID is bigger than 64bits, CmdTexture::mTextureId needs to be updated to support it");
	TextureCastHelperUnion textureUnion;
	textureUnion.TexClientID = 0;
	if( textureRef._TexData ){
		//Note: Cannot rely on textureRef.GetTexID() because it uses a pointer to
		//		a texture object that might not have been created by the backend yet.
		//		Instead, use a stable value that remain valid for texture lifetime,
		//		to id it with the NetImgui Server
		textureUnion.TexClientID = static_cast<ClientTextureID>(textureRef._TexData->UniqueID);
	}
	else{
		textureUnion.TexID 	= textureRef._TexID;
	}
	return textureUnion.TexClientID;
}
#endif

ClientTextureID ConvertToClientTexID(ImTextureID textureID)
{
	static_assert(sizeof(uint64_t) >= sizeof(ImTextureID), "ImTextureID is bigger than 64bits, CmdTexture::mTextureId needs to be updated to support it");
	TextureCastHelperUnion textureUnion;
	textureUnion.TexClientID	= 0;
	textureUnion.TexID			= textureID;
	return textureUnion.TexClientID;
	
}

ImTextureID ConvertFromClientTexID(ClientTextureID clientTexID)
{
	TextureCastHelperUnion textureUnion;
	textureUnion.TexClientID = clientTexID;
	return textureUnion.TexID;
}

}} //namespace NetImgui::Internal
