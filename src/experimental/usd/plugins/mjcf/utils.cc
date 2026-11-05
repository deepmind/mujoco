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

#include "mjcf/utils.h"

#include <string>

#include <pxr/base/tf/token.h>
#include <pxr/usd/sdf/attributeSpec.h>
#include <pxr/usd/sdf/declareHandles.h>
#include <pxr/usd/sdf/layer.h>
#include <pxr/usd/sdf/listOp.h>
#include <pxr/usd/sdf/path.h>
#include <pxr/usd/sdf/primSpec.h>
#include <pxr/usd/sdf/reference.h>
#include <pxr/usd/sdf/relationshipSpec.h>
#include <pxr/usd/sdf/schema.h>
#include <pxr/usd/sdf/types.h>
#include <pxr/usd/sdf/valueTypeName.h>
#include <pxr/usd/usd/tokens.h>
#include <pxr/usd/usdGeom/tokens.h>
namespace {

template <typename T>
void AppendListOp(pxr::SdfLayerRefPtr layer,
                  const pxr::SdfPrimSpecHandle& prim_spec, const pxr::TfToken& field,
                  const T& item) {
  pxr::SdfListOp<T> listOp =
      prim_spec->GetInfo(field).Get<pxr::SdfListOp<T>>();
  auto items = listOp.GetExplicitItems();
  items.push_back(item);
  listOp.SetExplicitItems(items);
  prim_spec->SetInfo(field, pxr::VtValue::Take(listOp));
}

template <typename T>
void PrependListOp(pxr::SdfLayerRefPtr layer,
                   const pxr::SdfPrimSpecHandle& prim_spec, const pxr::TfToken& field,
                   const T& item) {
  pxr::SdfListOp<T> listOp =
      prim_spec->GetInfo(field).Get<pxr::SdfListOp<T>>();
  auto prependedItems = listOp.GetPrependedItems();
  prependedItems.insert(prependedItems.begin(), item);
  listOp.SetPrependedItems(prependedItems);
  prim_spec->SetInfo(field, pxr::VtValue::Take(listOp));
}
}  // namespace

namespace mujoco {
namespace usd {

pxr::SdfPrimSpecHandle CreatePrimSpec(pxr::SdfLayerRefPtr layer,
                            const pxr::SdfPath& parent_path,
                            const pxr::TfToken& name, const pxr::TfToken& type,
                            pxr::SdfSpecifier specifier) {
  const pxr::SdfPath prim_path = parent_path.AppendChild(name);
  pxr::SdfPrimSpecHandle prim_spec = pxr::SdfCreatePrimInLayer(layer, prim_path);
  layer->SetField(prim_path, pxr::SdfFieldKeys->Specifier, specifier);
  if (!type.IsEmpty()) {
    layer->SetField(prim_path, pxr::SdfFieldKeys->TypeName, type);
  }

  return prim_spec;
}

pxr::SdfAttributeSpecHandle CreateAttributeSpec(pxr::SdfLayerRefPtr layer,
                                 const pxr::SdfPrimSpecHandle& prim_spec,
                                 const pxr::TfToken& name,
                                 const pxr::SdfValueTypeName& type_name,
                                 pxr::SdfVariability variability) {
  const pxr::SdfPath propertyPath = prim_spec->GetPath().AppendProperty(name);

  // Early exit if the attribute spec already exists.
  if (layer->HasSpec(propertyPath)) {
    return layer->GetAttributeAtPath(propertyPath);
  }

  auto spec = pxr::SdfAttributeSpec::New(prim_spec,
      name, type_name, variability);

  return spec;
}

pxr::SdfRelationshipSpecHandle CreateRelationshipSpec(pxr::SdfLayerRefPtr layer,
                                    const pxr::SdfPrimSpecHandle& prim_spec,
                                    const pxr::TfToken& relationship_name,
                                    const pxr::SdfPath& relationship_path,
                                    pxr::SdfVariability variability) {
  const pxr::SdfPath property_path =
      prim_spec->GetPath().AppendProperty(relationship_name);

  pxr::SdfRelationshipSpecHandle spec;
  if (layer->HasSpec(property_path)) {
    spec = layer->GetRelationshipAtPath(property_path);
  } else {
    spec = pxr::SdfRelationshipSpec::New(prim_spec, relationship_name,
                                         /*custom=*/true, variability);
  }
  if (spec) {
    spec->GetTargetPathList().Append(relationship_path);
  }

  return spec;
}

pxr::SdfPrimSpecHandle CreateClassSpec(pxr::SdfLayerRefPtr layer,
                             const pxr::SdfPath& prim_path,
                             const pxr::TfToken& class_name) {
  pxr::SdfPath class_path = prim_path.AppendChild(class_name);
  pxr::SdfPrimSpecHandle prim_spec = pxr::SdfCreatePrimInLayer(layer, class_path);
  layer->SetField(class_path, pxr::SdfFieldKeys->Specifier, pxr::SdfSpecifier::SdfSpecifierClass);

  return prim_spec;
}

void AddAttributeConnection(pxr::SdfLayerRefPtr layer,
                            const pxr::SdfAttributeSpecHandle& attribute_spec,
                            const pxr::SdfAttributeSpecHandle& target_attribute_spec) {
  attribute_spec->GetConnectionPathList().GetExplicitItems() = {target_attribute_spec->GetPath()};
}

void AddPrimReference(pxr::SdfLayerRefPtr layer,
                      const pxr::SdfPrimSpecHandle& prim_spec,
                      const pxr::SdfPath& referenced_prim_path) {
  PrependListOp(layer, prim_spec, pxr::SdfFieldKeys->References,
                pxr::SdfReference("", referenced_prim_path));
}

void AddPrimInherit(pxr::SdfLayerRefPtr layer,
                    const pxr::SdfPrimSpecHandle& prim_spec,
                    const pxr::SdfPath& class_path) {
  PrependListOp(layer, prim_spec, pxr::SdfFieldKeys->InheritPaths, class_path);
}

void ApplyApiSchema(pxr::SdfLayerRefPtr layer,
                    const pxr::SdfPrimSpecHandle& prim_spec,
                    const pxr::TfToken& schema_name) {
  PrependListOp(layer, prim_spec, pxr::UsdTokens->apiSchemas, schema_name);
}

void SetPrimKind(pxr::SdfLayerRefPtr layer,
                 const pxr::SdfPrimSpecHandle& prim_spec, pxr::TfToken kind) {
  prim_spec->SetField(pxr::SdfFieldKeys->Kind, kind);
}

void SetPrimPurpose(pxr::SdfLayerRefPtr layer,
                    const pxr::SdfPrimSpecHandle& prim_spec, pxr::TfToken purpose) {
  const pxr::SdfAttributeSpecHandle& attr = CreateAttributeSpec(
      layer, prim_spec, pxr::UsdGeomTokens->purpose,
      pxr::SdfValueTypeNames->Token, pxr::SdfVariabilityUniform);
  SetAttributeDefault(layer, attr, purpose);
}

}  // namespace usd
}  // namespace mujoco
