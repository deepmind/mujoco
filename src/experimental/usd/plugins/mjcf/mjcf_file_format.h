// Copyright 2025 DeepMind Technologies Limited
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

#ifndef MUJOCO_SRC_EXPERIMENTAL_USD_PLUGINS_MJCF_MJCF_FILE_FORMAT_H_
#define MUJOCO_SRC_EXPERIMENTAL_USD_PLUGINS_MJCF_MJCF_FILE_FORMAT_H_

#include <string>

#include <mujoco/mujoco.h>
#include <pxr/base/tf/declarePtrs.h>
#include <pxr/base/tf/staticTokens.h>
#include <pxr/pxr.h>
#include <pxr/usd/sdf/declareHandles.h>
#include <pxr/usd/sdf/fileFormat.h>

PXR_NAMESPACE_OPEN_SCOPE

// clang-format off
// The Id should realistically be mjcf, but the id and extension need to match.
// So near term it just assumes the only .xml file we would import is MJCF.
#define USD_MJCF_FILE_FORMAT_TOKENS  \
    ((Id, "xml"))                    \
    ((Version, "1.0"))               \
    ((Target, "usd"))                \
// clang-format on

TF_DECLARE_PUBLIC_TOKENS(UsdMjcfFileFormatTokens, USD_MJCF_FILE_FORMAT_TOKENS);

TF_DECLARE_WEAK_AND_REF_PTRS(UsdMjcfFileFormat);

class UsdMjcfFileFormat : public SdfFileFormat {
 public:
  using SdfFileFormat::FileFormatArguments;

  // Returns true if 'file' can be read by this format plugin.
  bool CanRead(const std::string &file) const override;

  // Reads scene description from the asset specified by resolved_path into
  // 'layer'.
  //
  // metadataOnly is a flag that asks for only the layer metadata to be read in,
  // which can be much faster if that is all that is required but currently we
  // ignore it.
  //
  // Returns true if the asset is successfully read into layer, false otherwise.
  bool Read(pxr::SdfLayer *layer, const std::string &resolved_path,
            bool metadata_only) const override;

  // Reads data in the string 'str' into 'layer'.
  //
  // If the file is successfully read, this method returns true. Otherwise,
  // false is returned and errors are posted.
  bool ReadFromString(SdfLayer *layer, const std::string &str) const override;

  // Writes the contents in 'layer' to 'str'. This just forwards to the usda
  // implementation.
  bool WriteToString(const SdfLayer &layer, std::string *str,
                     const std::string &comment) const override;

 protected:
  SDF_FILE_FORMAT_FACTORY_ACCESS;

  UsdMjcfFileFormat();
  virtual ~UsdMjcfFileFormat();

 private:
  // Function delegated to by Read and ReadFromString.
  bool ReadImpl(SdfLayer *layer, mjSpec *spec) const;
};

PXR_NAMESPACE_CLOSE_SCOPE

#endif  // MUJOCO_SRC_EXPERIMENTAL_USD_PLUGINS_MJCF_MJCF_FILE_FORMAT_H_
