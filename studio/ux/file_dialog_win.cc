// Copyright 2026 DeepMind Technologies Limited
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include "studio/ux/file_dialog.h"

#ifndef UNICODE
  #define UNICODE
#endif

#define _CRTDBG_MAP_ALLOC
#include <assert.h>
#include <crtdbg.h>
#include <shobjidl.h>
#include <stdio.h>
#include <stdlib.h>
#include <wchar.h>
#include <windows.h>

#include <functional>
#include <memory>
#include <span>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

namespace mujoco::studio {

static DialogResult SetError(std::string err) {
  return {.status = DialogResult::kError, .path = std::move(err)};
}

static std::string WCharToChar(const wchar_t* str) {
  int wlen = static_cast<int>(wcslen(str));
  int num_bytes = WideCharToMultiByte(CP_UTF8, 0, str, wlen, NULL, 0, NULL, NULL);
  assert(num_bytes);
  num_bytes += 1;

  std::string out;
  out.resize(num_bytes);
  WideCharToMultiByte(CP_UTF8, 0, str, -1, out.data(), num_bytes, NULL, NULL);
  return out;
}

static std::unique_ptr<wchar_t[]> CharToWChar(std::string_view str) {
  int len = static_cast<int>(str.size());
  int num_wchars = MultiByteToWideChar(CP_UTF8, 0, str.data(), len, NULL, 0);
  assert(num_wchars);
  num_wchars += 1;  // terminator

  wchar_t* ptr = new wchar_t[num_wchars];
  MultiByteToWideChar(CP_UTF8, 0, str.data(), num_wchars, ptr, num_wchars);
  ptr[num_wchars - 1] = '\0';
  return std::unique_ptr<wchar_t[]>(ptr);
}

static std::unique_ptr<wchar_t[]> ReadFilter(std::string_view filter) {
  std::string filter_str = FormatFileFilter(filter, ';');
  if (filter_str.empty()) {
    return nullptr;
  }
  return CharToWChar(filter_str);
}

static bool AddFilters(IFileDialog* dialog, std::span<std::string_view> filters = {}) {
  // +1 to include *.* wildcard.
  const size_t max_filters = filters.size() + 1;

  std::vector<COMDLG_FILTERSPEC> specs;
  std::vector<std::unique_ptr<wchar_t[]>> names;
  names.reserve(max_filters);
  specs.reserve(max_filters);

  for (std::string_view filter : filters) {
    auto wfilter = ReadFilter(filter);
    if (!wfilter) {
      continue;
    }
    auto& spec = specs.emplace_back();
    spec.pszName = wfilter.get();
    spec.pszSpec = wfilter.get();
    names.push_back(std::move(wfilter));
  }

  const wchar_t WILDCARD[] = L"*.*";
  auto& last = specs.emplace_back();
  last.pszSpec = WILDCARD;
  last.pszName = WILDCARD;

  dialog->SetFileTypes(static_cast<UINT>(specs.size()), specs.data());
  return true;
}

static bool SetDefaultPath(IFileDialog* dialog, std::string_view path) {
  if (path.empty() || path.front() == 0) {
    return true;
  }

  auto wpath = CharToWChar(path);

  IShellItem* folder(NULL);
  HRESULT hresult = SHCreateItemFromParsingName(wpath.get(), NULL, IID_PPV_ARGS(&folder));
  if (hresult == HRESULT_FROM_WIN32(ERROR_FILE_NOT_FOUND) ||
      hresult == HRESULT_FROM_WIN32(ERROR_INVALID_DRIVE)) {
    return true;
  }
  if (SUCCEEDED(hresult)) {
    dialog->SetFolder(folder);
    folder->Release();
    return true;
  }

  return false;
}

template <typename T>
static std::unique_ptr<T, std::function<void(T*)>> MakeDialog(REFCLSID class_id, REFIID ref_id) {
  T* ptr = nullptr;
  HRESULT hresult =
      ::CoCreateInstance(class_id, NULL, CLSCTX_ALL, ref_id, reinterpret_cast<void**>(&ptr));
  if (!SUCCEEDED(hresult) || ptr == nullptr) {
    return nullptr;
  }
  return {ptr, [](T* dialog) { dialog->Release(); }};
}

static DialogResult RunDialog(IFileDialog* dialog, std::string_view default_path,
                              std::span<std::string_view> filters = {}) {
  if (!dialog) {
    return SetError("Could not create dialog.");
  }
  if (!SetDefaultPath(dialog, default_path)) {
    return SetError("Could not set path.");
  }
  if (!AddFilters(dialog, filters)) {
    return SetError("Could not create dialog.");
  }

  HRESULT hresult = dialog->Show(nullptr);

  if (SUCCEEDED(hresult)) {
    IShellItem* item = nullptr;
    hresult = dialog->GetResult(&item);
    if (!SUCCEEDED(hresult)) {
      item->Release();
      return SetError("Could not get file path for selected.");
    }

    wchar_t* wselected = nullptr;
    hresult = item->GetDisplayName(SIGDN_DESKTOPABSOLUTEPARSING, &wselected);
    if (!SUCCEEDED(hresult)) {
      item->Release();
      return SetError("GetDisplayName for IShellItem failed.");
    }

    std::string selected_path = WCharToChar(wselected);
    CoTaskMemFree(wselected);
    item->Release();

    if (selected_path.empty()) {
      return SetError("");

    } else {
      return {.status = DialogResult::kAccepted, .path = std::move(selected_path)};
    }

  } else if (hresult == HRESULT_FROM_WIN32(ERROR_CANCELLED)) {
    return {.status = DialogResult::kCancelled};

  } else {
    return SetError("File dialog box show failed.");
  }
}

namespace {
struct ScopedCom {
  ScopedCom() {
    hresult = ::CoInitializeEx(NULL, ::COINIT_APARTMENTTHREADED | ::COINIT_DISABLE_OLE1DDE);
  }

  ~ScopedCom() {
    if (SUCCEEDED(hresult)) {
      ::CoUninitialize();
    }
  }

  bool ok() {
    if (hresult == RPC_E_CHANGED_MODE) {
      return true;
    }
    return SUCCEEDED(hresult);
  }

  HRESULT hresult;
};
}  // namespace

DialogResult OpenFileDialog(std::string_view path, std::span<std::string_view> filters) {
  ScopedCom com;
  if (!com.ok()) {
    return SetError("Could not initialize COM.");
  }

  auto dialog = MakeDialog<IFileOpenDialog>(CLSID_FileOpenDialog, IID_IFileOpenDialog);
  return RunDialog(dialog.get(), path, filters);
}

DialogResult SaveFileDialog(std::string_view path, std::span<std::string_view> filters) {
  ScopedCom com;
  if (!com.ok()) {
    return SetError("Could not initialize COM.");
  }

  auto dialog = MakeDialog<IFileSaveDialog>(CLSID_FileSaveDialog, IID_IFileSaveDialog);
  return RunDialog(dialog.get(), path, filters);
}

DialogResult SelectPathDialog(std::string_view path) {
  ScopedCom com;
  if (!com.ok()) {
    return SetError("Could not initialize COM.");
  }

  auto dialog = MakeDialog<IFileOpenDialog>(CLSID_FileOpenDialog, IID_IFileOpenDialog);
  DWORD options = 0;
  if (!SUCCEEDED(dialog->GetOptions(&options))) {
    return SetError("GetOptions for IFileDialog failed.");
  }
  if (!SUCCEEDED(dialog->SetOptions(options | FOS_PICKFOLDERS))) {
    return SetError("SetOptions for IFileDialog failed.");
  }
  return RunDialog(dialog.get(), path);
}
}  // namespace mujoco::studio
