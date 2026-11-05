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


#ifndef MUJOCO_SRC_XML_XML_UTIL_H_
#define MUJOCO_SRC_XML_XML_UTIL_H_

#include <array>
#include <functional>
#include <optional>
#include <set>
#include <string>
#include <vector>
#include <sstream>

#include <mujoco/mujoco.h>
#include "user/user_util.h"
#include "tinyxml2.h"

// error string copy
void mjCopyError(char* dst, const char* src, int maxlen);

using tinyxml2::XMLElement;

static constexpr int kDefaultUnauthoredValue = -12345;

XMLElement* FirstChildElement(XMLElement* e, const char* name = nullptr);
XMLElement* NextSiblingElement(XMLElement* e, const char* name = nullptr);

// XML Error info
class [[nodiscard]] mjXError {
 public:
  mjXError(const tinyxml2::XMLElement* elem = 0,
           const char*                 msg  = 0,
           const char*                 str  = 0,
           int                         pos  = 0);
  ~mjXError() = default;

  char message[1000];  // error message
};


// presence constraint over an element's attributes, generated from the
// schema into the grammar table's companion array. spec holds attribute
// bundles, space-joined within a bundle and '|'-separated between them;
// kind: e=exclusive (at most one bundle present), t=together (all or
// none), r=requires (first needs second), o=oneof (at least one bundle
// complete).
struct mjXConstraintDef {
  int         row;  // index of the element's row in the grammar table
  char        kind;
  const char* spec;
};


// Custom XML file validation
class mjXSchema {
 public:
  mjXSchema(std::vector<const char*> schema[],
            unsigned                 nrow,
            const mjXConstraintDef*  constraints = nullptr,
            int                      nconstraint = 0,
            int                      first_row   = 0);

  std::string GetError();                                      // return error
  void        Print(std::stringstream& str, int level) const;  // print schema
  void        PrintHTML(std::stringstream& str, int level, bool pad) const;

  bool                  NameMatch(tinyxml2::XMLElement* elem, int level);  // does name match
  tinyxml2::XMLElement* Check(tinyxml2::XMLElement* elem, int level);      // validator


 private:
  std::string            name_;       // element name
  char                   type_;       // element type: '?', '!', '*', 'R'
  std::set<std::string>  attr_;       // allowed attributes
  std::vector<mjXSchema> subschema_;  // allowed child elements

  std::vector<const mjXConstraintDef*> constraints_;  // constraints here
  tinyxml2::XMLElement*                CheckConstraints(tinyxml2::XMLElement* elem);

  int         refcnt_ = 0;  // refcount used for validation
  std::string error;        // error from constructor or Check
};


// key(string) : value(int) map
struct _mjMap {
  const char* key;
  int         value;
};
typedef struct _mjMap mjMap;


// XML read and write utility functions
class mjXUtil {
 public:
  mjXUtil()          = default;
  virtual ~mjXUtil() = default;

  // compare two vectors
  template <typename T>
  static bool SameVector(const T* vec1, const T* vec2, int n);

  // find key in map, return value (-1: not found)
  static int FindKey(const mjMap* map, int mapsz, std::string key);

  // find value in map, return key ("": not found)
  static std::string FindValue(const mjMap* map, int mapsz, int value);

  // if attribute is present, return vector of numerical data
  template <typename T>
  static std::optional<std::vector<T>> ReadAttrVec(tinyxml2::XMLElement* elem,
                                                   const char*           attr,
                                                   bool                  required = false);

  // if attribute is present, return attribute as a string
  static std::optional<std::string> ReadAttrStr(tinyxml2::XMLElement* elem,
                                                const char*           attr,
                                                bool                  required = false);

  // if attribute is present, return attribute as a filename
  static std::optional<mujoco::user::FilePath> ReadAttrFile(
      tinyxml2::XMLElement*         elem,
      const char*                   attr,
      const mjVFS*                  vfs,
      const mujoco::user::FilePath& dir      = mujoco::user::FilePath(),
      bool                          required = false);

  // if attribute is present, return numerical value of attribute
  template <typename T>
  static std::optional<T> ReadAttrNum(tinyxml2::XMLElement* elem,
                                      const char*           attr,
                                      bool                  required = false);

  // if attribute is present, return array of numerical data
  // N should be small as data is allocated on the stack
  template <typename T, int N>
  static std::optional<std::array<T, N>> ReadAttrArr(tinyxml2::XMLElement* elem,
                                                     const char*           attr,
                                                     bool                  required = false) {
    std::array<T, N> arr;
    int              n = 0;
    if (!ReadAttrValues<T>(
            elem,
            attr,
            [&](int i, T num) {
              arr[i] = num;
              n++;
            },
            N)) {
      throw mjXError(elem, "attribute '%s' has too much data", attr);
    }

    if (!n) {
      if (required) {
        throw mjXError(elem, "required attribute missing: '%s'", attr);
      } else {
        return std::nullopt;
      }
    }

    if (n < N) { throw mjXError(elem, "attribute '%s' does not have enough data", attr); }
    return arr;
  }

  // deprecated: use ReadAttrVec or ReadAttrArr
  template <typename T>
  static int ReadAttr(tinyxml2::XMLElement* elem,
                      const char*           attr,
                      int                   len,
                      T*                    data,
                      std::string&          text,
                      bool                  required = false,
                      bool                  exact    = true);

  static int ReadQuat(tinyxml2::XMLElement* elem,
                      const char*           attr,
                      double*               data,
                      std::string&          text,
                      bool                  required = false);

  // deprecated: use ReadAttrVec
  static int ReadVector(tinyxml2::XMLElement* elem,
                        const char*           attr,
                        std::vector<double>&  vec,
                        std::string&          text,
                        bool                  required = false);

  // deprecated: use ReadAttrStr
  static bool ReadAttrTxt(tinyxml2::XMLElement* elem,
                          const char*           attr,
                          std::string&          text,
                          bool                  required = false);

  // deprecated: use ReadAttrNum
  static bool ReadAttrInt(tinyxml2::XMLElement* elem,
                          const char*           attr,
                          int*                  data,
                          bool                  required = false);

  // write vector of numbers to string, as WriteAttr writes them
  template <typename T>
  static void Vector2String(std::string& txt, const std::vector<T>& vec, int ncol = 0);


  // find subelement with given name, make sure it is unique
  static tinyxml2::XMLElement* FindSubElem(tinyxml2::XMLElement* elem,
                                           std::string           name,
                                           bool                  required = false);

  // find attribute, translate key, return int value
  static bool MapValue(tinyxml2::XMLElement* elem,
                       const char*           attr,
                       int*                  data,
                       const mjMap*          map,
                       int                   mapSz,
                       bool                  required = false);

  template <typename T>
  static bool MapValue(tinyxml2::XMLElement* elem,
                       const char*           attr,
                       T*                    data,
                       const mjMap*          map,
                       int                   mapSz,
                       bool                  required = false) {
    int value;
    if (MapValue(elem, attr, &value, map, mapSz, required)) {
      *data = static_cast<T>(value);
      return true;
    }
    return false;
  }

  // find attribute, translate unique space-separated keys to data, return number of keys found
  static int MapValues(tinyxml2::XMLElement* elem,
                       const char*           attr,
                       int*                  data,
                       const mjMap*          map,
                       int                   mapSz,
                       bool                  required = false);

  // write attribute- any type
  template <typename T>
  static void WriteAttr(tinyxml2::XMLElement* elem,
                        std::string           name,
                        int                   n,
                        const T*              data,
                        const T*              def  = 0,
                        bool                  trim = false);

  // write vector<double> attribute, with and without default
  static void WriteVector(tinyxml2::XMLElement*      elem,
                          std::string                name,
                          const std::vector<double>& vec);
  static void WriteVector(tinyxml2::XMLElement*      elem,
                          std::string                name,
                          const std::vector<double>& vec,
                          const std::vector<double>& def);

  // write attribute- string
  static void WriteAttrTxt(tinyxml2::XMLElement* elem, std::string name, std::string value);

  // write attribute- single int
  static void WriteAttrInt(tinyxml2::XMLElement* elem,
                           std::string           name,
                           int                   data,
                           int                   def = kDefaultUnauthoredValue);

  // write attribute- keyword
  static void WriteAttrKey(tinyxml2::XMLElement* elem,
                           std::string           name,
                           const mjMap*          map,
                           int                   mapsz,
                           int                   data,
                           int                   def = kDefaultUnauthoredValue);

  // write attribute- space-separated keywords
  static void WriteAttrKeys(XMLElement*  elem,
                            std::string  name,
                            const mjMap* map,
                            int          mapsz,
                            int*         data,
                            int          ndata,
                            int          def = kDefaultUnauthoredValue);

 private:
  template <typename T>
  static bool ReadAttrValues(tinyxml2::XMLElement*       elem,
                             const char*                 attr,
                             std::function<void(int, T)> push,
                             int                         max = -1);
};

#endif  // MUJOCO_SRC_XML_XML_UTIL_H_
