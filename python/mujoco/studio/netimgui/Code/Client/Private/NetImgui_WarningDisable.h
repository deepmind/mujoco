#pragma once
//
// Deactivate a few warnings to allow internal netImgui code to compile
// with 'Warning as error' and '-Wall' compile actions enabled
//

//=================================================================================================
// Clang
//=================================================================================================
#if defined(__clang__)
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wunknown-warning-option"
#pragma clang diagnostic ignored "-Wc++98-compat-pedantic"
#pragma clang diagnostic ignored "-Wmissing-prototypes"
#pragma clang diagnostic ignored "-Wold-style-cast"  // For ImTextureID_Invalid
#pragma clang diagnostic ignored "-Wunsafe-buffer-usage"
#pragma clang diagnostic ignored "-Wswitch-default"
#pragma clang diagnostic ignored \
    "-Wnontrivial-memcall"  // For ImGui::IO memcpy warning

//=================================================================================================
// Visual Studio warnings
//=================================================================================================
#elif defined(_MSC_VER)
#pragma warning(disable : 5032)  // detected #pragma warning(push) with no
                                 // corresponding #pragma warning(pop)

#pragma warning(push)
#pragma warning(disable : 4365)  // conversion from 'long' to 'unsigned int',
                                 // signed/unsigned mismatch for <atomic>
#pragma warning(disable : 4464)  // relative include path contains '..'
#pragma warning(disable \
                : 4514)  // unreferenced inline function has been removed
#pragma warning( \
    disable      \
    : 4577)  // 'noexcept' used with no exception handling mode specified;
             // termination on exception is not guaranteed. Specify
#pragma warning(disable : 4710)  // 'xxx': function not inlined
#pragma warning( \
    disable : 4711)  // function 'xxx' selected for automatic inline expansion
#pragma warning( \
    disable      \
    : 4826)  // Conversion from 'TType *' to 'uint64_t' is sign-extended. This
             // may cause unexpected runtime behavior.
#pragma warning(disable \
                : 5031)  // #pragma warning(pop): likely mismatch, popping
                         // warning state pushed in different file
#pragma warning(disable : 5045)  // Compiler will insert Spectre mitigation for
                                 // memory load if / Qspectre switch specified
#pragma warning(disable : 5264)  // 'xxx': 'const' variable is not used

#endif
