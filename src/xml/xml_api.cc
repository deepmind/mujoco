// Copyright 2021 DeepMind Technologies Limited
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

#include "xml/xml_api.h"

#include <array>
#include <cstdio>
#include <cstring>
#include <functional>
#include <memory>
#include <sstream>
#include <string>

#include <mujoco/mjmodel.h>
#include <mujoco/mjspec.h>
#include "engine/engine_io.h"
#include "engine/engine_util_errmem.h"
#include "user/user_resource.h"
#include "xml/xml.h"
#include "xml/xml_global.h"
#include "xml/xml_native_reader.h"
#include "xml/xml_util.h"

//---------------------------------- Functions -----------------------------------------------------

// parse XML file in MJCF or URDF format, compile it, return low-level model
//  if vfs is not NULL, look up files in vfs before reading from disk
//  error can be NULL; otherwise assumed to have size error_sz
mjModel* mj_loadXML(const char* filename, const mjVFS* vfs, char* error, int error_sz) {
  // parse new model
  std::unique_ptr<mjSpec, std::function<void(mjSpec*)> > spec(
      ParseXML(filename, vfs, error, error_sz),
      [](mjSpec* s) { mj_deleteSpec(s); });
  if (!spec) { return nullptr; }

  // compile new model
  mjModel* m = mj_compile(spec.get(), vfs);
  if (!m) {
    mjCopyError(error, mjs_getError(spec.get()), error_sz);
    return nullptr;
  }

  // handle compile warning
  int num_warnings = mjs_numWarnings(spec.get());
  if (num_warnings > 0) {
    std::string all_warnings;
    for (int i = 0; i < num_warnings; ++i) {
      if (!all_warnings.empty()) { all_warnings += '\n'; }
      all_warnings += mjs_getWarning(spec.get(), i);
    }
    mjCopyError(error, all_warnings.c_str(), error_sz);
  } else if (error) {
    error[0] = '\0';
  }

  // clear old and assign new
  SetGlobalXmlSpec(spec.release());
  return m;
}


// update XML data structures with info from low-level model, save as MJCF
//  returns 1 if successful, 0 otherwise
//  error can be NULL; otherwise assumed to have size error_sz
int mj_saveLastXML(const char* filename, const mjModel* m, char* error, int error_sz) {
  // the file is opened once there is something to write: a failure leaves it as it was
  const std::string result = GetGlobalXmlSpec(m, error, error_sz);
  if (result.empty()) { return 0; }

  FILE* fp = stdout;
  if (filename != nullptr && filename[0] != '\0') {
    fp = mju_fopen(filename, "w");
    if (!fp) {
      mjCopyError(error, "File not found", error_sz);
      return 0;
    }
  }

  fprintf(fp, "%s", result.c_str());
  if (fp != stdout) { fclose(fp); }

  return 1;
}


// free last XML
void mj_freeLastXML(void) {
  SetGlobalXmlSpec();
}


// print internal XML schema as plain text or HTML, with style-padding or &nbsp;
int mj_printSchema(const char* filename, char* buffer, int buffer_sz, int flg_html, int flg_pad) {
  // print to stringstream
  mjXReader         reader;
  std::stringstream str;
  reader.PrintSchema(str, flg_html != 0, flg_pad != 0);

  // filename given: write to file
  if (filename) {
    FILE* file = mju_fopen(filename, "w");
    if (file) {
      fputs(str.str().c_str(), file);
      fclose(file);
    }
  }

  // buffer given: write to buffer
  if (buffer && buffer_sz) {
    strncpy(buffer, str.str().c_str(), buffer_sz);
    buffer[buffer_sz - 1] = 0;
  }

  // return string length
  return str.str().size();
}


// load model from binary MJB resource
mjModel* mj_loadModel(const char* filename, const mjVFS* vfs) {
  std::array<char, 1024> error;

  mjResource* resource = mju_openResource("", filename, vfs, error.data(), error.size());
  if (resource == nullptr) {
    mju_warning("%s", error.data());
    return nullptr;
  }

  const void* buffer    = NULL;
  int         buffer_sz = mju_readResource(resource, &buffer);
  if (buffer_sz < 1) {
    mju_closeResource(resource);
    return nullptr;
  }

  mjModel* m = mj_loadModelBuffer(buffer, buffer_sz);
  mju_closeResource(resource);
  return m;
}


// parse spec from file
mjSpec* mj_parseXML(const char* filename, const mjVFS* vfs, char* error, int error_sz) {
  return ParseXML(filename, vfs, error, error_sz);
}


// parse spec from string
mjSpec* mj_parseXMLString(const char* xml, const mjVFS* vfs, char* error, int error_sz) {
  return ParseSpecFromString(xml, vfs, error, error_sz);
}


// save spec to XML file, return 0 on success, -1 otherwise
int mj_saveXML(const mjSpec* s, const char* filename, char* error, int error_sz) {
  // cast to mjSpec since WriteXML can in principle perform mj_copyBack (not here)
  std::string result = WriteXML(NULL, (mjSpec*)s, error, error_sz);
  if (result.empty()) { return -1; }

  mjtSize written =
      mju_writeResource(filename, result.data(), result.size(), NULL, error, error_sz);
  if (written != result.size()) {
    if (error && error_sz > 0 && error[0] == '\0') {
      std::snprintf(error, error_sz, "Error writing XML file '%s'", filename);
    }
    return -1;
  }
  return 0;
}


// save spec to XML string, return 0 on success, -1 on failure
// if length of the output buffer is too small, returns the required size
int mj_saveXMLString(const mjSpec* s, char* xml, int xml_sz, char* error, int error_sz) {
  std::string result = WriteXML(NULL, (mjSpec*)s, error, error_sz);
  if (result.empty()) {
    return -1;
  } else if (result.size() >= xml_sz) {
    std::string error_msg =
        "Output string too short, should be at least " + std::to_string(result.size() + 1);
    mjCopyError(error, error_msg.c_str(), error_sz);
    return result.size();
  }

  result.copy(xml, xml_sz);
  xml[result.size()] = 0;
  return 0;
}
