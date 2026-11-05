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

#include "studio/ux/gui_spec.h"

#include <string>
#include <string_view>

#include <imgui.h>
#include <mujoco/mjxmacro.h>
#include <mujoco/mujoco.h>
#include "studio/ux/gui.h"
#include "studio/ux/imgui_widgets.h"
#include "studio/ux/spec_editor.h"

// Define the mujoco X macros to add fields to the ImGui_DataTable.
// We limit the fields to the ones with a matching element by comparing the
// array size field (e.g. nbody) with the MATCH constexpr value.
#define X(TYPE, NAME, NELEM, SIZE) \
  if constexpr (#NELEM == MATCH) table.DataPtr(#NAME, ptr->NAME, index, SIZE);

// Simple wrapper around MJMODEL_POINTERS that prepares values we need for the
// X macro above.
#define MJMODEL_POINTERS_X(M)           \
  constexpr std::string_view MATCH(#M); \
  const auto* ptr = model;              \
  MJMODEL_POINTERS

// Simple wrapper around MJDATA_POINTERS that prepares values we need for the
// X macro above.
#define MJDATA_POINTERS_X(M)            \
  constexpr std::string_view MATCH(#M); \
  const auto* ptr = data;               \
  MJDATA_POINTERS

namespace mujoco::studio {

// Returns the index of the element in the spec. This is different from
// mjs_getId which returns the runtime ID of an element.
static int GetElementIndexInSpec(mjsElement* element) {
  int n = 0;
  mjSpec* spec = mjs_getSpec(element);
  mjsElement* iter = mjs_firstElement(spec, element->elemtype);
  while (iter) {
    if (iter == element) {
      return n;
    }
    iter = mjs_nextElement(spec, iter);
    ++n;
  }
  return -1;
}

// Returns a name for the element; either the element has a name, or we
// construct a unique name from the element's id (using mjs_getId) or index
// (using GetElementIndexInSpec).
static std::string GetElementName(mjsElement* element) {
  const mjString* name = mjs_getName(element);
  std::string label = *name;
  if (label.empty()) {
    int id = mjs_getId(element);
    if (id == -1) {
      id = GetElementIndexInSpec(element);
    }
    const char* type_name = mju_type2Str(element->elemtype);
    label = "(" + std::string(type_name) + " " + std::to_string(id) + ")";
  }
  return label;
}

static void AddEditorButtons(mjsElement* element, mjsElement** selected_element,
                             SpecEditor& editor) {
  // Right-align the buttons.
  float x = ImGui::GetWindowContentRegionMax().x;
  x -= ImGui::GetStyle().FramePadding.x * 2;
  x -= ImGui::CalcTextSize(ICON_FA_TRASH_CAN).x;
  if (element->elemtype == mjOBJ_BODY) {
    x -= ImGui::GetStyle().FramePadding.x * 2;
    x -= ImGui::CalcTextSize(ICON_FA_PLUS).x;
  }
  ImGui::SameLine(x);
  if (element->elemtype == mjOBJ_BODY) {
    if (ImGui::SmallButton(ICON_FA_PLUS)) {
      ImGui::OpenPopupOnItemClick("BodyAddChild", 0);
    }

    if (ImGui::BeginPopupContextItem("BodyAddChild")) {
      mjsBody* body = mjs_asBody(element);
      auto option = [&](const char* label, mjtObj type) {
        if (ImGui::Selectable(label)) {
          *selected_element = editor.AddBodyElement(body, type);
        }
      };
      option("Camera", mjOBJ_CAMERA);
      option("Frame", mjOBJ_FRAME);
      option("Geom", mjOBJ_GEOM);
      option("Joint", mjOBJ_JOINT);
      option("Light", mjOBJ_LIGHT);
      option("Site", mjOBJ_SITE);
      ImGui::EndPopup();
    }

    ImGui::SameLine();
  }
  if (ImGui::SmallButton(ICON_FA_TRASH_CAN)) {
    editor.DeleteActiveElement();
    *selected_element = nullptr;
  }
}

static void SelectableElement(mjsElement* element, mjsElement** selected_element,
                              SpecEditor* editor) {
  constexpr ImGuiSelectableFlags flags = ImGuiSelectableFlags_AllowOverlap;

  const std::string name = GetElementName(element);
  const bool selected = (element == *selected_element);
  if (ImGui::Selectable(name.c_str(), selected, flags)) {
    *selected_element = element;
  }
  if (selected && editor) {
    AddEditorButtons(element, selected_element, *editor);
  }
}

static void BodyChildrenGui(const char* heading, mjtObj type, mjsElement** element, mjsBody* body,
                            SpecEditor* editor) {
  mjsElement* iter = mjs_firstChild(body, type, 0);
  if (!iter) {
    return;
  }

  constexpr ImGuiTreeNodeFlags tree_flags =
      ImGuiTreeNodeFlags_SpanAvailWidth | ImGuiTreeNodeFlags_DrawLinesFull;
  if (SectionHeader(heading, tree_flags, 0.55f)) {
    while (iter) {
      mjsElement* next = mjs_nextChild(body, iter, 0);
      SelectableElement(iter, element, editor);
      iter = next;
    }
    ImGui::TreePop();
  }
}

static void ElementListGui(const char* heading, mjtObj type, mjsElement** element, mjSpec* spec,
                           SpecEditor* editor) {
  mjsElement* iter = mjs_firstElement(spec, type);
  if (!iter) {
    return;
  }

  constexpr ImGuiTreeNodeFlags tree_flags =
      ImGuiTreeNodeFlags_SpanAvailWidth | ImGuiTreeNodeFlags_Framed;
  if (SectionHeader(heading, tree_flags, 0.55f)) {
    while (iter) {
      mjsElement* next = mjs_nextElement(spec, iter);
      SelectableElement(iter, element, editor);
      iter = next;
    }
    ImGui::TreePop();
  }
}

static void BodyTreeGuiRecursive(mjsElement** element, mjsBody* body, SpecEditor* editor) {
  const std::string label = GetElementName(body->element);

  ImGui::PushID(label.c_str());

  ImGuiTreeNodeFlags flags = ImGuiTreeNodeFlags_SpanAvailWidth | ImGuiTreeNodeFlags_Framed |
                             ImGuiTreeNodeFlags_DrawLinesFull | ImGuiTreeNodeFlags_AllowOverlap;

  if (*element == body->element) {
    flags |= ImGuiTreeNodeFlags_Selected;
  }

  const bool tree_open = SectionHeader(label.c_str(), flags, 0.55f);
  if (ImGui::IsItemClicked()) {
    *element = body->element;
  }
  if (editor && *element == body->element) {
    AddEditorButtons(body->element, element, *editor);
    if (!*element) {
      if (tree_open) {
        ImGui::TreePop();
      }
      ImGui::PopID();
      return;
    }
  }

  if (tree_open) {
    mjsElement* iter = mjs_firstChild(body, mjOBJ_BODY, 0);
    while (iter) {
      mjsElement* next = mjs_nextChild(body, iter, 0);
      BodyTreeGuiRecursive(element, mjs_asBody(iter), editor);
      iter = next;
    }

    BodyChildrenGui("Frames", mjOBJ_FRAME, element, body, editor);
    BodyChildrenGui("Sites", mjOBJ_SITE, element, body, editor);
    BodyChildrenGui("Joints", mjOBJ_JOINT, element, body, editor);
    BodyChildrenGui("Geoms", mjOBJ_GEOM, element, body, editor);
    BodyChildrenGui("Lights", mjOBJ_LIGHT, element, body, editor);
    BodyChildrenGui("Cameras", mjOBJ_CAMERA, element, body, editor);
    ImGui::TreePop();
  }
  ImGui::PopID();
}

void SpecTreeGui(mjsElement** element, mjSpec* spec, SpecEditor* editor) {
  const ImGuiTreeNodeFlags flags = ImGuiTreeNodeFlags_SpanAvailWidth | ImGuiTreeNodeFlags_Framed;

  ScopedStyle style;
  style.Var(ImGuiStyleVar_ItemSpacing, ImVec2(4, 2));
  style.Var(ImGuiStyleVar_FramePadding, ImVec2(4, 0));
  style.Color(ImGuiCol_Border, ImGuiCol_WindowBg);

  if (SectionHeader("Body Tree", flags, 0.65f)) {
    mjsElement* root = mjs_firstElement(spec, mjOBJ_BODY);
    if (root) {
      mjsBody* body = mjs_asBody(root);
      if (body) {
        BodyTreeGuiRecursive(element, body, editor);
      }
    }
    ImGui::TreePop();
  }

  auto list = [&](const char* heading, mjtObj type) {
    ElementListGui(heading, type, element, spec, editor);
  };

  ImGui::PushID("$spec$");

  // Non-tree elements.
  if (SectionHeader("Elements", flags, 0.65f)) {
    list("Actuators", mjOBJ_ACTUATOR);
    list("Sensors", mjOBJ_SENSOR);
    list("Flexes", mjOBJ_FLEX);
    list("Tendons", mjOBJ_TENDON);
    list("Pair", mjOBJ_PAIR);
    list("Exclude", mjOBJ_EXCLUDE);
    list("Equality", mjOBJ_EQUALITY);
    list("Numeric", mjOBJ_NUMERIC);
    list("Text", mjOBJ_TEXT);
    list("Tuple", mjOBJ_TUPLE);
    list("Key", mjOBJ_KEY);
    list("Default", mjOBJ_DEFAULT);
    ImGui::TreePop();
  }

  // Assets.
  if (SectionHeader("Assets", flags, 0.65f)) {
    list("Meshes", mjOBJ_MESH);
    list("Height Fields", mjOBJ_HFIELD);
    list("Skins", mjOBJ_SKIN);
    list("Textures", mjOBJ_TEXTURE);
    list("Materials", mjOBJ_MATERIAL);
    ImGui::TreePop();
  }

  ImGui::PopID();
}

void ElementSpecGui(mjsElement* element, SpecEditor* editor) {
  if (element == nullptr) {
    return;
  }

  // if editor, assert that element == editor->GetActiveElement();
  mjsElement* ref_element = editor ? editor->GetRefElement() : element;

#define FIELD(NAME, TIP) table(#NAME, elem->NAME, ref->NAME, TIP);
#define QFIELD(NAME, ALT, TIP) table(#NAME, #ALT, elem->NAME, ref->NAME, elem->ALT, ref->ALT, TIP);

  ImGui_SpecElementTable table(editor == nullptr);
  switch (element->elemtype) {
    case mjOBJ_BODY: {
      mjsBody* elem = mjs_asBody(element);
      mjsBody* ref = mjs_asBody(ref_element);
      FIELD(childclass, "childclass name");
      FIELD(pos, "frame position");
      QFIELD(quat, alt, "frame orientation");
      FIELD(ipos, "inertial frame position");
      QFIELD(iquat, ialt, "inertial frame orientation");
      FIELD(mass, "mass");
      FIELD(inertia, "diagonal inertia (in i-frame)");
      FIELD(fullinertia, "non-axis-aligned inertia matrix");
      FIELD(mocap, "is this a mocap body");
      FIELD(gravcomp, "gravity compensation");
      FIELD(explicitinertial, "whether to save the body with explicit inertial clause");
      FIELD(sleep, "sleep policy");
      FIELD(info, "message appended to compiler errors");
      break;
    }
    case mjOBJ_JOINT: {
      mjsJoint* elem = mjs_asJoint(element);
      mjsJoint* ref = mjs_asJoint(ref_element);
      FIELD(pos, "anchor position");
      FIELD(axis, "joint axis");
      FIELD(ref, "value at reference configuration: qpos0");
      FIELD(align, "align free joint with body com (mjtAlignFree)");
      FIELD(stiffness, "stiffness coefficient");
      FIELD(springref, "spring reference value: qpos_spring");
      FIELD(springdamper, "timeconst, dampratio");
      FIELD(springyield, "yield threshold for plastic spring");
      FIELD(limited, "does joint have limits (mjtLimited)");
      FIELD(range, "joint limits");
      FIELD(margin, "margin value for joint limit detection");
      FIELD(solref_limit, "solver reference: joint limits");
      FIELD(solimp_limit, "solver impedance: joint limits");
      FIELD(actfrclimited, "are actuator forces on joint limited (mjtLimited)");
      FIELD(actfrcrange, "actuator force limits");
      FIELD(armature, "armature inertia (mass for slider)");
      FIELD(damping, "damping coefficient");
      FIELD(frictionloss, "friction loss");
      FIELD(solref_friction, "solver reference: dof friction");
      FIELD(solimp_friction, "solver impedance: dof friction");
      FIELD(group, "group");
      FIELD(actgravcomp, "is gravcomp force applied via actuators");
      FIELD(info, "message appended to compiler errors");
      break;
    }
    case mjOBJ_ACTUATOR: {
      mjsActuator* elem = mjs_asActuator(element);
      mjsActuator* ref = mjs_asActuator(ref_element);
      FIELD(gaintype, "gain type");
      FIELD(gainprm, "gain parameters");
      FIELD(biastype, "bias type");
      FIELD(biasprm, "bias parameters");
      FIELD(dyntype, "dynamics type");
      FIELD(dynprm, "dynamics parameters");
      FIELD(actdim, "number of activation variables");
      FIELD(actearly, "apply next activations to qfrc");
      FIELD(trntype, "transmission type");
      FIELD(gear, "length and transmitted force scaling");
      FIELD(target, "name of transmission target");
      FIELD(refsite, "reference site, for site transmission");
      FIELD(slidersite, "site defining cylinder, for slider-crank");
      FIELD(cranklength, "crank length, for slider-crank");
      FIELD(lengthrange, "transmission length range");
      FIELD(inheritrange, "automatic range setting for position and intvelocity");
      FIELD(ctrllimited, "are control limits defined (mjtLimited)");
      FIELD(ctrlrange, "control range");
      FIELD(forcelimited, "are force limits defined (mjtLimited)");
      FIELD(forcerange, "force range");
      FIELD(actlimited, "are activation limits defined (mjtLimited)");
      FIELD(actrange, "activation range");
      FIELD(group, "group");
      FIELD(nsample, "number of samples in history buffer");
      FIELD(interp, "interpolation order (0=ZOH, 1=linear, 2=cubic)");
      FIELD(delay, "delay time in seconds; 0: no delay");
      FIELD(info, "message appended to compiler errors");
      break;
    }
    case mjOBJ_SENSOR: {
      mjsSensor* elem = mjs_asSensor(element);
      mjsSensor* ref = mjs_asSensor(ref_element);
      FIELD(type, "type of sensor");
      FIELD(objtype, "type of sensorized object");
      FIELD(objname, "name of sensorized object");
      FIELD(reftype, "type of referenced object");
      FIELD(refname, "name of referenced object");
      FIELD(intprm, "integer parameters");
      FIELD(datatype, "data type for sensor measurement");
      FIELD(needstage, "compute stage needed to simulate sensor");
      FIELD(dim, "number of scalar outputs");
      FIELD(cutoff, "cutoff for real and positive datatypes");
      FIELD(noise, "noise stdev");
      FIELD(nsample, "number of samples in history buffer");
      FIELD(interp, "interpolation order (0=ZOH, 1=linear, 2=cubic)");
      FIELD(delay, "delay time in seconds");
      FIELD(interval, "[period, time_prev] in seconds");
      FIELD(info, "message appended to compiler errors");
      break;
    }
    case mjOBJ_SITE: {
      mjsSite* elem = mjs_asSite(element);
      mjsSite* ref = mjs_asSite(ref_element);
      FIELD(pos, "position");
      QFIELD(quat, alt, "orientation");
      FIELD(fromto, "alternative for capsule, cylinder, box, ellipsoid");
      FIELD(size, "geom size");
      FIELD(type, "geom type");
      FIELD(material, "name of material");
      FIELD(group, "group");
      FIELD(rgba, "rgba when material is omitted");
      FIELD(info, "message appended to compiler errors");
      break;
    }
    case mjOBJ_FRAME: {
      mjsFrame* elem = mjs_asFrame(element);
      mjsFrame* ref = mjs_asFrame(ref_element);
      FIELD(childclass, "childclass name");
      FIELD(pos, "position");
      QFIELD(quat, alt, "orientation");
      FIELD(info, "message appended to compiler errors");
      break;
    }
    case mjOBJ_GEOM: {
      mjsGeom* elem = mjs_asGeom(element);
      mjsGeom* ref = mjs_asGeom(ref_element);
      FIELD(type, "geom type");
      FIELD(pos, "position");
      QFIELD(quat, alt, "orientation");
      FIELD(fromto, "alternative for capsule, cylinder, box, ellipsoid");
      FIELD(size, "type-specific size");
      FIELD(contype, "contact type");
      FIELD(conaffinity, "contact affinity");
      FIELD(condim, "contact dimensionality");
      FIELD(priority, "contact priority");
      FIELD(friction, "one-sided friction coefficients: slide, roll, spin");
      FIELD(solmix, "solver mixing for contact pairs");
      FIELD(solref, "solver reference");
      FIELD(solimp, "solver impedance");
      FIELD(margin, "margin for contact detection");
      FIELD(gap, "additional contact detection buffer");
      FIELD(mass, "used to compute density");
      FIELD(density, "used to compute mass and inertia from volume or surface");
      FIELD(typeinertia, "selects between surface and volume inertia");
      FIELD(fluid_ellipsoid, "whether ellipsoid-fluid model is active");
      FIELD(fluid_coefs, "ellipsoid-fluid interaction coefs");
      FIELD(material, "name of material");
      FIELD(rgba, "rgba when material is omitted");
      FIELD(group, "group");
      FIELD(castshadow, "does geom cast shadows");
      FIELD(hfieldname, "heightfield attached to geom");
      FIELD(meshname, "mesh attached to geom");
      FIELD(fitscale, "scale mesh uniformly");
      FIELD(info, "message appended to compiler errors");
      break;
    }
    case mjOBJ_LIGHT: {
      mjsLight* elem = mjs_asLight(element);
      mjsLight* ref = mjs_asLight(ref_element);
      FIELD(pos, "position");
      FIELD(dir, "direction");
      FIELD(mode, "tracking mode");
      FIELD(targetbody, "target body for targeting");
      FIELD(active, "is light active");
      FIELD(type, "type of light");
      FIELD(texture, "texture name for image lights");
      FIELD(castshadow, "does light cast shadows");
      FIELD(bulbradius, "bulb radius, for soft shadows");
      FIELD(intensity, "intensity, in candelas");
      FIELD(range, "range of effectiveness");
      FIELD(attenuation, "OpenGL attenuation (quadratic model)");
      FIELD(cutoff, "OpenGL cutoff");
      FIELD(exponent, "OpenGL exponent");
      FIELD(ambient, "ambient color");
      FIELD(diffuse, "diffuse color");
      FIELD(specular, "specular color");
      FIELD(info, "message appended to compiler errorsx");
      break;
    }
    case mjOBJ_CAMERA: {
      mjsCamera* elem = mjs_asCamera(element);
      mjsCamera* ref = mjs_asCamera(ref_element);
      FIELD(pos, "position");
      QFIELD(quat, alt, "orientation");
      FIELD(mode, "tracking mode");
      FIELD(targetbody, "target body for tracking/targeting");
      FIELD(proj, "camera projection type");
      FIELD(resolution, "resolution (pixel)");
      FIELD(output, "bit flags for output type");
      FIELD(fovy, "y-field of view");
      FIELD(ipd, "inter-pupillary distance");
      FIELD(intrinsic, "camera intrinsics (length)");
      FIELD(sensor_size, "sensor size (length)");
      FIELD(focal_length, "focal length (length)");
      FIELD(focal_pixel, "focal length (pixel)");
      FIELD(principal_length, "principal point (length)");
      FIELD(principal_pixel, "principal point (pixel)");
      FIELD(info, "message appended to compiler errors");
      break;
    }
    case mjOBJ_MESH: {
      mjsMesh* elem = mjs_asMesh(element);
      mjsMesh* ref = mjs_asMesh(ref_element);
      FIELD(content_type, "content type of file");
      FIELD(file, "mesh file");
      FIELD(refpos, "reference position");
      FIELD(refquat, "reference orientation");
      FIELD(scale, "rescale mesh");
      FIELD(inertia, "inertia type (convex, legacy, exact, shell)");
      FIELD(smoothnormal, "do not exclude large-angle faces from normals");
      FIELD(needsdf, "compute sdf from mesh");
      FIELD(maxhullvert, "maximum vertex count for the convex hull");
      FIELD(material, "name of material");
      FIELD(info, "message appended to compiler errors");
      break;
    }
    case mjOBJ_HFIELD: {
      mjsHField* elem = mjs_asHField(element);
      mjsHField* ref = mjs_asHField(ref_element);
      FIELD(content_type, "content type of file");
      FIELD(file, "file: (nrow, ncol, [elevation data])");
      FIELD(size, "hfield size (ignore referencing geom size)");
      FIELD(nrow, "number of rows");
      FIELD(ncol, "number of columns");
      FIELD(info, "message appended to compiler errors");
      break;
    }
    case mjOBJ_SKIN: {
      mjsSkin* elem = mjs_asSkin(element);
      mjsSkin* ref = mjs_asSkin(ref_element);
      FIELD(file, "skin file");
      FIELD(material, "name of material used for rendering");
      FIELD(rgba, "rgba when material is omitted");
      FIELD(inflate, "inflate in normal direction");
      FIELD(group, "group for visualization");
      FIELD(info, "message appended to compiler errors");
      break;
    }
    case mjOBJ_FLEX: {
      mjsFlex* elem = mjs_asFlex(element);
      mjsFlex* ref = mjs_asFlex(ref_element);
      FIELD(contype, "contact type");
      FIELD(conaffinity, "contact affinity");
      FIELD(condim, "contact dimensionality");
      FIELD(priority, "contact priority");
      FIELD(friction, "one-sided friction coefficients: slide, roll, spin");
      FIELD(solmix, "solver mixing for contact pairs");
      FIELD(solref, "solver reference");
      FIELD(solimp, "solver impedance");
      FIELD(margin, "margin for contact detection");
      FIELD(gap, "additional contact detection buffer");
      FIELD(dim, "element dimensionality");
      FIELD(radius, "radius around primitive element");
      FIELD(size, "vertex bounding box half sizes in qpos0");
      FIELD(flatskin, "render flex skin with flat shading");
      FIELD(selfcollide, "mode for flex self collision");
      FIELD(passive, "mode for passive collisions");
      FIELD(activelayers, "number of active element layers in 3D");
      FIELD(group, "group for visualization");
      FIELD(edgestiffness, "edge stiffness");
      FIELD(edgedamping, "edge damping");
      FIELD(rgba, "rgba when material is omitted");
      FIELD(material, "name of material used for rendering");
      FIELD(young, "Young's modulus");
      FIELD(poisson, "Poisson's ratio");
      FIELD(shear, "1D torsion shear modulus; 0: young / (2 (1 + poisson))");
      FIELD(damping, "Rayleigh's damping");
      FIELD(thickness, "thickness (2D only)");
      FIELD(stressyield, "yield strain thresholds (stretch, bend); 0: elastic");
      FIELD(elastic2d, "2D passive forces; 0: none, 1: bending, 2: stretching, 3: both");
      FIELD(section, "1D cross-section (mjtFlexSection)");
      FIELD(sectionsize, "1D box cross-section half-sizes");
      FIELD(info, "message appended to compiler errors");
      break;
    }
    case mjOBJ_TENDON: {
      mjsTendon* elem = mjs_asTendon(element);
      mjsTendon* ref = mjs_asTendon(ref_element);
      FIELD(stiffness, "stiffness coefficient");
      FIELD(springlength, "spring resting length; {-1, -1}: use qpos_spring");
      FIELD(springyield, "yield threshold for plastic spring");
      FIELD(damping, "damping coefficient");
      FIELD(frictionloss, "friction loss");
      FIELD(solref_friction, "solver reference: tendon friction");
      FIELD(solimp_friction, "solver impedance: tendon friction");
      FIELD(armature, "inertia associated with tendon velocity");
      FIELD(limited, "does tendon have limits (mjtLimited)");
      FIELD(actfrclimited, "does tendon have actuator force limits");
      FIELD(range, "length limits");
      FIELD(actfrcrange, "actuator force limits");
      FIELD(margin, "margin value for tendon limit detection");
      FIELD(solref_limit, "solver reference: tendon limits");
      FIELD(solimp_limit, "solver impedance: tendon limits");
      FIELD(material, "name of material for rendering");
      FIELD(width, "width for rendering");
      FIELD(rgba, "rgba when material is omitted");
      FIELD(group, "group");
      FIELD(info, "message appended to errors");
      break;
    }
    case mjOBJ_TEXTURE: {
      mjsTexture* elem = mjs_asTexture(element);
      mjsTexture* ref = mjs_asTexture(ref_element);
      FIELD(type, "texture type");
      FIELD(colorspace, "colorspace");
      FIELD(builtin, "builtin type (mjtBuiltin)");
      FIELD(mark, "mark type (mjtMark)");
      FIELD(rgb1, "first color for builtin");
      FIELD(rgb2, "second color for builtin");
      FIELD(markrgb, "mark color");
      FIELD(random, "probability of random dots");
      FIELD(height, "height in pixels (square for cube and skybox)");
      FIELD(width, "width in pixels");
      FIELD(nchannel, "number of channels");
      FIELD(content_type, "content type of file");
      FIELD(file, "png file to load; use for all sides of cube");
      FIELD(gridsize, "size of grid for composite file; (1,1)-repeat");
      FIELD(gridlayout, "row-major: L,R,F,B,U,D for faces; . for unused");
      FIELD(cubefiles, "different file for each side of the cube");
      FIELD(hflip, "horizontal flip");
      FIELD(vflip, "vertical flip");
      FIELD(info, "message appended to compiler errors");
      break;
    }
    case mjOBJ_MATERIAL: {
      mjsMaterial* elem = mjs_asMaterial(element);
      mjsMaterial* ref = mjs_asMaterial(ref_element);
      FIELD(textures, "names of textures (empty: none)");
      FIELD(texuniform, "make texture cube uniform");
      FIELD(texrepeat, "texture repetition for 2D mapping");
      FIELD(emission, "emission");
      FIELD(specular, "specular");
      FIELD(shininess, "shininess");
      FIELD(reflectance, "reflectance");
      FIELD(metallic, "metallic");
      FIELD(roughness, "roughness");
      FIELD(rgba, "rgba");
      FIELD(info, "message appended to compiler errors");
      break;
    }
    case mjOBJ_PAIR: {
      mjsPair* elem = mjs_asPair(element);
      mjsPair* ref = mjs_asPair(ref_element);
      FIELD(geomname1, "name of geom 1");
      FIELD(geomname2, "name of geom 2");
      FIELD(condim, "contact dimensionality");
      FIELD(solref, "solver reference, normal direction");
      FIELD(solreffriction, "solver reference, frictional directions");
      FIELD(solimp, "solver impedance");
      FIELD(margin, "margin for contact detection");
      FIELD(gap, "additional contact detection buffer");
      FIELD(friction, "full contact friction");
      FIELD(info, "message appended to errors");
      break;
    }
    case mjOBJ_EQUALITY: {
      mjsEquality* elem = mjs_asEquality(element);
      mjsEquality* ref = mjs_asEquality(ref_element);
      FIELD(type, "constraint type");
      FIELD(data, "type-dependent data");
      FIELD(active, "is equality initially active");
      FIELD(name1, "name of object 1");
      FIELD(name2, "name of object 2");
      FIELD(objtype, "type of both objects");
      FIELD(solref, "solver reference");
      FIELD(solimp, "solver impedance");
      FIELD(info, "message appended to errors");
      break;
    }
    case mjOBJ_EXCLUDE: {
      mjsExclude* elem = mjs_asExclude(element);
      mjsExclude* ref = mjs_asExclude(ref_element);
      FIELD(bodyname1, "name of geom 1");
      FIELD(bodyname2, "name of geom 2");
      FIELD(info, "message appended to errors");
      break;
    }
    case mjOBJ_NUMERIC: {
      mjsNumeric* elem = mjs_asNumeric(element);
      mjsNumeric* ref = mjs_asNumeric(ref_element);
      FIELD(data, "initialization data");
      FIELD(size, "array size, can be bigger than data size");
      FIELD(info, "message appended to errors");
      break;
    }
    case mjOBJ_TEXT: {
      mjsText* elem = mjs_asText(element);
      mjsText* ref = mjs_asText(ref_element);
      FIELD(data, "text string");
      FIELD(info, "message appended to compiler errors");
      break;
    }
    case mjOBJ_TUPLE: {
      mjsTuple* elem = mjs_asTuple(element);
      mjsTuple* ref = mjs_asTuple(ref_element);
      FIELD(objtype, "object types");
      FIELD(objname, "object names");
      FIELD(objprm, "object parameters");
      FIELD(info, "message appended to compiler errors");
      break;
    }
    case mjOBJ_KEY: {
      mjsKey* elem = mjs_asKey(element);
      mjsKey* ref = mjs_asKey(ref_element);
      FIELD(time, "time");
      FIELD(qpos, "qpos");
      FIELD(qvel, "qvel");
      FIELD(act, "act");
      FIELD(mpos, "mocap pos");
      FIELD(mquat, "mocap quat");
      FIELD(ctrl, "ctrl");
      FIELD(info, "message appended to compiler errors");
      break;
    }
    case mjOBJ_PLUGIN: {
      mjsPlugin* elem = mjs_asPlugin(element);
      mjsPlugin* ref = mjs_asPlugin(ref_element);
      FIELD(name, "instance name");
      FIELD(plugin_name, "plugin name");
      FIELD(active, "is the plugin active");
      FIELD(info, "message appended to compiler errors");
      break;
    }
    default:
      // ignore other types
      break;
  }

#undef FIELD

  if (editor && table.WasModified()) {
    editor->CommitChanges(element);
  }
}

void ElementModelGui(const mjModel* model, mjsElement* element) {
  if (element == nullptr) {
    return;
  }

  ImGui_DataPtrTable table;
  const int index = mjs_getId(element);

  MJMODEL_POINTERS_PREAMBLE(model);
  switch (element->elemtype) {
    case mjOBJ_BODY: {
      table.SetPrefix("body_");
      MJMODEL_POINTERS_X(nbody)
      break;
    }
    case mjOBJ_JOINT: {
      table.SetPrefix("jnt_");
      MJMODEL_POINTERS_X(njnt)
      break;
    }
    case mjOBJ_ACTUATOR: {
      table.SetPrefix("actuator_");
      MJMODEL_POINTERS_X(nu)
      break;
    }
    case mjOBJ_SENSOR: {
      table.SetPrefix("sensor_");
      MJMODEL_POINTERS_X(nsensor)
      break;
    }
    case mjOBJ_SITE: {
      table.SetPrefix("site_");
      MJMODEL_POINTERS_X(nsite)
      break;
    }
    case mjOBJ_GEOM: {
      table.SetPrefix("geom_");
      MJMODEL_POINTERS_X(ngeom)
      break;
    }
    case mjOBJ_LIGHT: {
      table.SetPrefix("light_");
      MJMODEL_POINTERS_X(nlight)
      break;
    }
    case mjOBJ_CAMERA: {
      table.SetPrefix("cam_");
      MJMODEL_POINTERS_X(ncam)
      break;
    }
    case mjOBJ_MESH: {
      table.SetPrefix("mesh_");
      MJMODEL_POINTERS_X(nmesh)
      break;
    }
    case mjOBJ_HFIELD: {
      table.SetPrefix("hfield_");
      MJMODEL_POINTERS_X(nhfield)
      break;
    }
    case mjOBJ_SKIN: {
      table.SetPrefix("skin_");
      MJMODEL_POINTERS_X(nskin)
      break;
    }
    case mjOBJ_FLEX: {
      table.SetPrefix("flex_");
      MJMODEL_POINTERS_X(nflex)
      break;
    }
    case mjOBJ_TENDON: {
      table.SetPrefix("tendon_");
      MJMODEL_POINTERS_X(ntendon)
      break;
    }
    case mjOBJ_TEXTURE: {
      table.SetPrefix("tex_");
      MJMODEL_POINTERS_X(ntex)
      break;
    }
    case mjOBJ_MATERIAL: {
      table.SetPrefix("mat_");
      MJMODEL_POINTERS_X(nmat)
      break;
    }
    default:
      // ignore other types
      break;
  }
}

void ElementDataGui(const mjData* data, mjsElement* element) {
  if (element == nullptr) {
    return;
  }

  ImGui_DataPtrTable table;
  const int index = mjs_getId(element);

  switch (element->elemtype) {
    case mjOBJ_BODY: {
      MJDATA_POINTERS_X(nbody)
      break;
    }
    case mjOBJ_JOINT: {
      MJDATA_POINTERS_X(njnt)
      break;
    }
    case mjOBJ_SITE: {
      table.SetPrefix("site_");
      MJDATA_POINTERS_X(nsite)
      break;
    }
    case mjOBJ_GEOM: {
      table.SetPrefix("geom_");
      MJDATA_POINTERS_X(ngeom)
      break;
    }
    case mjOBJ_CAMERA: {
      table.SetPrefix("cam_");
      MJDATA_POINTERS_X(ncam)
      break;
    }
    case mjOBJ_LIGHT: {
      table.SetPrefix("light_");
      MJDATA_POINTERS_X(nlight)
      break;
    }
    case mjOBJ_SENSOR: {
      table.SetPrefix("sensor_");
      MJDATA_POINTERS_X(nsensor)
      break;
    }
    case mjOBJ_MESH: {
      table.SetPrefix("mesh_");
      MJDATA_POINTERS_X(nmesh)
      break;
    }
    case mjOBJ_HFIELD: {
      table.SetPrefix("hfield_");
      MJDATA_POINTERS_X(nhfield)
      break;
    }
    case mjOBJ_SKIN: {
      table.SetPrefix("skin_");
      MJDATA_POINTERS_X(nskin)
      break;
    }
    case mjOBJ_FLEX: {
      table.SetPrefix("flex_");
      MJDATA_POINTERS_X(nflex)
      break;
    }
    case mjOBJ_TENDON: {
      table.SetPrefix("ten_");
      MJDATA_POINTERS_X(ntendon)
      break;
    }
    case mjOBJ_TEXTURE: {
      table.SetPrefix("tex_");
      MJDATA_POINTERS_X(ntex)
      break;
    }
    case mjOBJ_MATERIAL: {
      table.SetPrefix("mat_");
      MJDATA_POINTERS_X(nmat)
      break;
    }
    default:
      break;
  }
}
}  // namespace mujoco::studio
