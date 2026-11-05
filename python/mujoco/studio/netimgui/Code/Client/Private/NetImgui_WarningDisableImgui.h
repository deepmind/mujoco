#pragma once
//
// Deactivate a few warnings to allow Imgui header includes,
// without generating warnings in '-Wall' compile actions enabled
//

//=================================================================================================
// Clang
//=================================================================================================
#if defined(__clang__)
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wunknown-warning-option"
#pragma clang diagnostic ignored "-Wc++98-compat-pedantic"
#pragma clang diagnostic ignored \
    "-Wnonportable-include-path"  // Sharpmake convert include path to
                                  // lowercase, avoid warning
#pragma clang diagnostic ignored \
    "-Wreserved-identifier"  // Enum values using '__' or member starting with
                             // '_' in imgui.h

//=================================================================================================
// Visual Studio warnings
//=================================================================================================
#elif defined(_MSC_VER)
#pragma warning(push)
#pragma warning( \
    disable : 4514)  // 'xxx': unreferenced inline function has been removed
#pragma warning(disable : 4365)  // '=': conversion from 'ImGuiTabItemFlags' to
                                 // 'ImGuiID', signed/unsigned mismatch
#pragma warning(disable : 4710)  // 'xxx': function not inlined
#pragma warning( \
    disable      \
    : 4820)  // 'xxx': 'yyy' bytes padding added after data member 'zzz'
#pragma warning(disable \
                : 5031)  // #pragma warning(pop): likely mismatch, popping
                         // warning state pushed in different file
#pragma warning(disable : 5045)  // Compiler will insert Spectre mitigation for
                                 // memory load if /Qspectre switch specified
#if _MSC_VER >= 1920
#pragma warning(disable : 5219)  // implicit conversion from 'int' to 'float',
                                 // possible loss of data
#endif
#pragma warning( \
    disable      \
    : 26495)  // Code Analysis warning : Variable
              // 'ImGuiStorage::ImGuiStoragePair::<unnamed-tag>::val_p' is
              // uninitialized. Always initialize a member variable (type.6).
#endif
