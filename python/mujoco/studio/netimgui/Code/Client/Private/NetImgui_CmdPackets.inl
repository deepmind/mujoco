// Google modifications:
//  - ToPointers() of commands received from the network return false when
//    their offsets or sizes are invalid (added CmdTexture::ToPointers()). The
//    draw frame arrays must be ComDataType aligned and own their last element.
#include "NetImgui_CmdPackets.h"

namespace NetImgui { namespace Internal
{

bool CmdTexture::ToPointers()
{
	if( mSize < sizeof(CmdTexture) ){
		return false;
	}
	// Data must hold the pixels described by the command (none when destroying)
	const uint64_t pixelsSize = mStatus == eType::Destroy ? 0 : static_cast<uint64_t>(GetTexture_BytePerLine(static_cast<eTexFormat>(mFormat), mWidth)) * mHeight;
	return mpTextureData.ToPointer(this, reinterpret_cast<const uint8_t*>(this) + mSize, pixelsSize);
}

// The draw frame data is laid out in ComDataType elements (see
// SetAndIncreaseDataPointer() and CompressData()), and the delta codec reads it
// one whole element at a time: each array must start on an element boundary of
// the command, and own all the elements it spans.
inline bool IsComDataAligned(const void* pData, const void* pBufferStart)
{
	return (reinterpret_cast<uint64_t>(pData) - reinterpret_cast<uint64_t>(pBufferStart)) % ComDataSize == 0;
}

bool CmdDrawFrame::ToPointers()
{
	if( mSize < sizeof(CmdDrawFrame) ){
		return false;
	}
	const uint8_t* pBufferEnd	= reinterpret_cast<const uint8_t*>(this) + mSize;
	bool bValid					= mpDrawGroups.ToPointer(this, pBufferEnd, static_cast<uint64_t>(mDrawGroupCount) * sizeof(ImguiDrawGroup)) &&
								  IsComDataAligned(mpDrawGroups.Get(), this);
	uint64_t verticeCount(0), indiceCount(0), drawCount(0);
	for (uint32_t i(0); bValid && i < mDrawGroupCount; ++i) {
		bValid			= mpDrawGroups[i].ToPointers(this, pBufferEnd, mCompressed != 0);
		verticeCount	+= mpDrawGroups[i].mVerticeCount;
		indiceCount		+= mpDrawGroups[i].mIndiceCount;
		drawCount		+= mpDrawGroups[i].mDrawCount;
	}
	// Totals are used by the receiver to allocate the buffers combining all groups
	return bValid && verticeCount == mTotalVerticeCount && indiceCount == mTotalIndiceCount && drawCount == mTotalDrawCount;
}

void CmdDrawFrame::ToOffsets()
{
	if( !mpDrawGroups.IsOffset() )
	{
		for (uint32_t i(0); i < mDrawGroupCount; ++i) {
			mpDrawGroups[i].ToOffsets();
		}
		mpDrawGroups.ToOffset();
	}
}

bool ImguiDrawGroup::ToPointers(const void* pBufferStart, const void* pBufferEnd, bool bCompressed)
{
	// Compressed streams size is only known once decompressed, only validate their start
	const uint64_t indiceSize	= bCompressed ? 0 : RoundUp<uint64_t>(static_cast<uint64_t>(mIndiceCount) * mBytePerIndex, ComDataSize);
	const uint64_t verticeSize	= bCompressed ? 0 : RoundUp<uint64_t>(static_cast<uint64_t>(mVerticeCount) * sizeof(ImguiVert), ComDataSize);
	const uint64_t drawSize		= bCompressed ? 0 : RoundUp<uint64_t>(static_cast<uint64_t>(mDrawCount) * sizeof(ImguiDraw), ComDataSize);
	return	(mBytePerIndex == 2 || mBytePerIndex == 4) &&
			mpIndices.ToPointer(pBufferStart, pBufferEnd, indiceSize) && IsComDataAligned(mpIndices.Get(), pBufferStart) &&
			mpVertices.ToPointer(pBufferStart, pBufferEnd, verticeSize) && IsComDataAligned(mpVertices.Get(), pBufferStart) &&
			mpDraws.ToPointer(pBufferStart, pBufferEnd, drawSize) && IsComDataAligned(mpDraws.Get(), pBufferStart);
}

void ImguiDrawGroup::ToOffsets()
{
	if( !mpIndices.IsOffset() ) //Safer to test the first element after CmdHeader
	{
		mpIndices.ToOffset();
		mpVertices.ToOffset();
		mpDraws.ToOffset();
	}
}

bool CmdInput::IsKeyDown( CmdInput::NetImguiKeys netimguiKey) const
{
	uint32_t valIndex	= netimguiKey/64;
	uint64_t valMask	= 0x0000000000000001ull << (netimguiKey%64);
	return mInputDownMask[valIndex] & valMask;
}

bool CmdBackground::operator==(const CmdBackground& cmp)const
{
	bool sameValue(true);
	for(size_t i(0); i<sizeof(CmdBackground)/8; i++){
		sameValue &= reinterpret_cast<const uint64_t*>(this)[i] == reinterpret_cast<const uint64_t*>(&cmp)[i];
	}
	return sameValue;
}

bool CmdBackground::operator!=(const CmdBackground& cmp)const
{
	return (*this == cmp) == false;
}


bool CmdClipboard::ToPointers()
{
	if( mSize < sizeof(CmdClipboard) ){
		return false;
	}
	// Content must be inside the command and null terminated
	return	mByteSize > 0 &&
			mContentUTF8.ToPointer(this, reinterpret_cast<const uint8_t*>(this) + mSize, mByteSize) &&
			mContentUTF8[mByteSize-1] == 0;
}

void CmdClipboard::ToOffsets()
{
	if( !mContentUTF8.IsOffset() ){
		mContentUTF8.ToOffset();
	}
}

CmdClipboard* CmdClipboard::Create(const char* clipboard)
{
	if( clipboard )
	{
		size_t clipboardByteSize(0);
		while(clipboard[clipboardByteSize++] != 0);
		size_t totalDataCount			= sizeof(CmdClipboard) + DivUp<size_t>(clipboardByteSize, ComDataSize);
		auto pNewClipboard				= NetImgui::Internal::netImguiSizedNew<CmdClipboard>(totalDataCount*ComDataSize);
		pNewClipboard->mSize			= static_cast<uint32_t>(totalDataCount*ComDataSize);
		pNewClipboard->mByteSize		= clipboardByteSize;
		pNewClipboard->mContentUTF8.SetPtr(reinterpret_cast<char*>(&pNewClipboard[1]));
		memcpy(pNewClipboard->mContentUTF8.Get(), clipboard, clipboardByteSize);
		return pNewClipboard;
	}
	return nullptr;
}

}} // namespace NetImgui::Internal
