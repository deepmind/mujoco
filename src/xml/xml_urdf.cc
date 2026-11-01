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

#include <array>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <string>
#include <vector>

#include <mujoco/mjmodel.h>
#include <mujoco/mjspec.h>
#include <mujoco/mjtype.h>
#include <mujoco/mujoco.h>
#include "user/user_api.h"
#include "user/user_util.h"
#include "xml/xml_native_reader.h"
#include "xml/xml_urdf.h"
#include "xml/xml_util.h"
#include "tinyxml2.h"

using tinyxml2::XMLElement;

// URDF joint type
static const int   urJoint_sz              = 7;
static const mjMap urJoint_map[urJoint_sz] = {
    {"revolute",   0},
    {"continuous", 1},
    {"prismatic",  2},
    {"fixed",      3},
    {"floating",   4},
    {"planar",     5},
    {"spherical",  6}  // Bullet physics supports ball joints (non-standard URDF)
};


//---------------------------------- class mjXURDF -------------------------------------------------

// constructor
mjXURDF::mjXURDF() {
  Clear();
}


// destructor
mjXURDF::~mjXURDF() {
  Clear();
}


// clear internal variables
void mjXURDF::Clear(void) {
  spec = 0;

  urName.clear();
  urParent.clear();
  urChildren.clear();
  urMat.clear();
  urRGBA.clear();
  urGeomNames.clear();
  urMimic.clear();
}

std::string mjXURDF::GetPrefixedName(const std::string& name) {
  if (name.empty()) { return name; }
  if (urPrefix.empty()) { return name; }
  return urPrefix + "/" + name;
}

// actual parser
void mjXURDF::Parse(XMLElement*        root,
                    const std::string& prefix,
                    double*            pos,
                    double*            quat,
                    const bool         static_body) {
  std::string name, text;
  XMLElement *elem, *temp;
  int         id_parent, id_child;
  urPrefix = prefix;

  // parse MuJoCo sections (not part of URDF)
  XMLElement* mjc = FindSubElem(root, "mujoco");
  if (mjc) {
    XMLElement* section;
    if ((section = FindSubElem(mjc, "compiler"))) { mjXReader::Compiler(section, spec); }

    if ((section = FindSubElem(mjc, "option"))) { mjXReader::Option(section, spec, &spec->option); }

    if ((section = FindSubElem(mjc, "size"))) { mjXReader::Size(section, spec); }
  }

  // enforce required compiler defaults for URDF
  spec->compiler.degree = false;

  // get model name
  std::string modelname;
  if (ReadAttrTxt(root, "name", modelname)) { mjs_setString(spec->modelname, modelname.c_str()); }

  // find and register all materials
  MakeMaterials(root);

  // find all links/bodies, save names
  elem = root->FirstChildElement();
  while (elem) {
    // identify link elements
    name = elem->Value();
    if (name == "link") {
      ReadAttrTxt(elem, "name", text, true);
      text = GetPrefixedName(text);
      AddBody(text);
    }

    // advance to next element
    elem = elem->NextSiblingElement();
  }

  // find all joints, assign parent and child pointers
  elem = root->FirstChildElement();
  while (elem) {
    // identify joint elements
    name = elem->Value();
    if (name == "joint") {
      // find parent, get name and id
      temp = FindSubElem(elem, "parent", true);
      ReadAttrTxt(temp, "link", text, true);
      text      = GetPrefixedName(text);
      id_parent = FindName(text, urName);

      // find child, get name and id
      temp = FindSubElem(elem, "child", true);
      ReadAttrTxt(temp, "link", text, true);
      text     = GetPrefixedName(text);
      id_child = FindName(text, urName);

      // make sure parent and child exist
      if (id_parent < 0 || id_child < 0) {
        throw mjXError(elem, "URDF joint parent or child missing");
      }

      // check for multiple parents
      if (urParent[id_child] >= 0) {
        throw mjXError(elem, "URDF body has multiple parents:", urName[id_child].c_str());
      }

      // add parent and child info
      urParent[id_child] = id_parent;
      urChildren[id_parent].push_back(id_child);
    }

    // advance to next element
    elem = elem->NextSiblingElement();
  }

  // find all top-level bodies, call recursive tree constructor
  for (int i = 0; i < (int)urName.size(); i++) {
    if (urParent[i] < 0) { AddToTree(i); }
  }

  // parse bodies
  elem = root->FirstChildElement();
  while (elem) {
    // identify body/link elements
    name = elem->Value();
    if (name == "link") { Body(elem); }

    // advance to next element
    elem = elem->NextSiblingElement();
  }

  // parse joints
  elem = root->FirstChildElement();
  while (elem) {
    // identify body/link elements
    name = elem->Value();
    if (name == "joint") { Joint(elem); }

    // advance to next element
    elem = elem->NextSiblingElement();
  }

  // <mimic> -> joint equality: q_follower - q0 = offset + multiplier*(q_driver - q0)
  auto hinge_or_slide = [&](const std::string& jnt_name) {
    mjsElement* el  = mjs_findElement(spec, mjOBJ_JOINT, jnt_name.c_str());
    mjsJoint*   jnt = el ? mjs_asJoint(el) : nullptr;
    return jnt && (jnt->type == mjJNT_HINGE || jnt->type == mjJNT_SLIDE);
  };
  for (const auto& [follower, driver, multiplier, offset] : urMimic) {
    // mjEQ_JOINT couples two 1-dof joints; skip the mimic rather than fail the
    // load when the target is fixed, planar, floating or missing
    if (!hinge_or_slide(follower) || !hinge_or_slide(driver)) {
      mju_warning(
          "<mimic> on joint '%s' ignored; mimic requires revolute, continuous, or prismatic"
          " joints",
          follower.c_str());
      continue;
    }
    mjsEquality* eq = mjs_addEquality(spec, nullptr);
    eq->type        = mjEQ_JOINT;
    mjs_setString(eq->name1, follower.c_str());
    mjs_setString(eq->name2, driver.c_str());
    eq->data[0] = offset;
    eq->data[1] = multiplier;
  }

  // override the pose for the base link and add a free joint
  for (int i = 0; i < (int)urName.size(); i++) {
    if (urParent[i] < 0) {
      mjsBody* world = mjs_findBody(spec, "world");
      mjsBody* pbody = mjs_findChild(world, urName[i].c_str());
      mjuu_copyvec(pbody->pos, pos, 3);
      mjuu_copyvec(pbody->quat, quat, 4);

      // add a free joint to allow motion of the body
      // if the mass is 0, assume the object is static
      if (!static_body && pbody->mass > 0) {
        mjsJoint* pjoint = mjs_addJoint(pbody, 0);
        mjs_setName(pjoint->element, (urName[i] + "_free_joint").c_str());
        pjoint->type = mjJNT_FREE;
      }
    }
  }
}

// parse body/link
void mjXURDF::Body(XMLElement* body_elem) {
  std::string name, text;
  XMLElement *elem, *temp, *temp1;
  mjsBody *   pbody, *world;
  mjsGeom*    pgeom;

  // get body name and pointer to mjsBody
  ReadAttrTxt(body_elem, "name", name, true);
  name  = GetPrefixedName(name);
  world = mjs_findBody(spec, "world");
  pbody = mjs_findChild(world, name.c_str());
  if (!pbody) {
    throw mjXError(body_elem, "URDF body not found");  // SHOULD NOT OCCUR
  }
  // inertial element: copy into alternative body frame
  if ((elem = FindSubElem(body_elem, "inertial"))) {
    pbody->explicitinertial = true;
    // origin- relative to joint frame for now
    Origin(elem, pbody->ipos, pbody->iquat);

    // mass
    temp = FindSubElem(elem, "mass", true);
    ReadAttr(temp, "value", 1, &pbody->mass, text, true);

    // inertia
    temp = FindSubElem(elem, "inertia", true);
    ReadAttr(temp, "ixx", 1, pbody->fullinertia + 0, text, true);
    ReadAttr(temp, "iyy", 1, pbody->fullinertia + 1, text, true);
    ReadAttr(temp, "izz", 1, pbody->fullinertia + 2, text, true);
    ReadAttr(temp, "ixy", 1, pbody->fullinertia + 3, text, true);
    ReadAttr(temp, "ixz", 1, pbody->fullinertia + 4, text, true);
    ReadAttr(temp, "iyz", 1, pbody->fullinertia + 5, text, true);

    // If the inertias are all 0 in a URDF then it is still undefined.
    bool inertia_defined = false;
    for (int i = 0; i < 6; ++i) {
      if (pbody->fullinertia[i] != 0) {
        inertia_defined = true;
        break;
      }
    }
    if (!inertia_defined) { pbody->fullinertia[0] = mjNAN; }
  }

  // clear body frame; set by joint later
  mjuu_setvec(pbody->pos, 0, 0, 0);
  mjuu_setvec(pbody->quat, 1, 0, 0, 0);

  // process all visual and geometry elements in order
  float rgba[4] = {-1, 0, 0, 0};

  elem = body_elem->FirstChildElement();
  while (elem) {
    std::string geom_name;
    name = elem->Value();

    // visual element
    if (name == "visual") {
      // parse material
      if ((temp = FindSubElem(elem, "material"))) {
        // if color specified - use directly
        if ((temp1 = FindSubElem(temp, "color"))) {
          ReadAttr(temp1, "rgba", 4, rgba, text, /*required=*/true);
        }

        // otherwise use material table
        else {
          std::string mat_name;
          ReadAttrTxt(temp, "name", mat_name, true);
          int imat = FindName(mat_name, urMat);
          if (imat >= 0) { std::memcpy(rgba, urRGBA[imat].val, 4 * sizeof(float)); }
        }
      }
      // create geom if not discarded
      if (!spec->compiler.discardvisual) {
        pgeom = Geom(elem, pbody, false);

        // save color
        if (rgba[0] >= 0) { std::memcpy(pgeom->rgba, rgba, 4 * sizeof(float)); }

        // save name if it doesn't already exist.
        mjXUtil::ReadAttrTxt(elem, "name", geom_name);
        geom_name = GetPrefixedName(geom_name);
        if (urGeomNames.find(geom_name) == urGeomNames.end()) {
          mjs_setName(pgeom->element, geom_name.c_str());
          urGeomNames.insert(geom_name);
        } else if (!geom_name.empty()) {
          std::cerr << "WARNING: Geom with duplicate name '" << geom_name
                    << "' encountered in URDF, creating an unnamed geom." << std::endl;
        }
      }
    }

    // collision element
    else if (name == "collision") {
      pgeom = Geom(elem, pbody, true);

      // use color from last visual
      if (rgba[0] >= 0) { std::memcpy(pgeom->rgba, rgba, 4 * sizeof(float)); }

      // save name if it doesn't already exist.
      mjXUtil::ReadAttrTxt(elem, "name", geom_name);
      geom_name = GetPrefixedName(geom_name);
      if (urGeomNames.find(geom_name) == urGeomNames.end()) {
        mjs_setName(pgeom->element, geom_name.c_str());
        urGeomNames.insert(geom_name);
      } else if (!geom_name.empty()) {
        std::cerr << "WARNING: Geom with duplicate name '" << geom_name
                  << "' encountered in URDF, creating an unnamed geom." << std::endl;
      }
    }
    // advance
    elem = elem->NextSiblingElement();
  }
}

void mjXURDF::Parse(XMLElement* root, const mjVFS* vfs) {
  double pos[3] = {0};
  mjuu_setvec(pos, 0, 0, 0);
  double quat[4] = {1, 0, 0, 0};
  mjuu_setvec(quat, 1, 0, 0, 0);
  Parse(root, /*prefix=*/"", pos, quat, true);
}

// parse joint
void mjXURDF::Joint(XMLElement* joint_elem) {
  std::string jntname, name, text;
  XMLElement* elem;
  mjsBody*    pbody;
  mjsBody*    parent;
  mjsBody*    world;
  mjsJoint*   pjoint  = 0;
  mjsJoint*   pjoint1 = 0;
  mjsJoint*   pjoint2 = 0;
  int         jointtype;

  // get type and name
  ReadAttrTxt(joint_elem, "type", text, true);
  jointtype = FindKey(urJoint_map, urJoint_sz, text);
  if (jointtype < 0) { throw mjXError(joint_elem, "invalid joint type in URDF joint definition"); }
  ReadAttrTxt(joint_elem, "name", jntname, true);
  jntname = GetPrefixedName(jntname);
  // get parent, check
  elem = FindSubElem(joint_elem, "parent", true);
  ReadAttrTxt(elem, "link", name, true);
  name   = GetPrefixedName(name);
  world  = mjs_findBody(spec, "world");
  parent = mjs_findChild(world, name.c_str());
  if (!parent) {  // SHOULD NOT OCCUR
    throw mjXError(elem, "invalid parent name in URDF joint definition");
  }

  // get child=this, check
  elem = FindSubElem(joint_elem, "child", true);
  ReadAttrTxt(elem, "link", name, true);
  name  = GetPrefixedName(name);
  world = mjs_findBody(spec, "world");
  pbody = mjs_findChild(world, name.c_str());
  if (!pbody) {  // SHOULD NOT OCCUR
    throw mjXError(elem, "invalid child name in URDF joint definition");
  }

  // read origin and axis
  double axis[3] = {1, 0, 0};
  Origin(joint_elem, pbody->pos, pbody->quat);
  if ((elem = FindSubElem(joint_elem, "axis"))) {
    ReadAttr(elem, "xyz", 3, axis, text, /*required=*/true);
  }

  // create joint (unless fixed)
  double mat[9], quat[4], tmpaxis[3];
  switch (jointtype) {
    case 0:  // revolute
    case 1:  // continuous
      pjoint = mjs_addJoint(pbody, 0);
      mjs_setName(pjoint->element, jntname.c_str());
      pjoint->type = mjJNT_HINGE;
      mjuu_setvec(pjoint->pos, 0, 0, 0);
      mjuu_copyvec(pjoint->axis, axis, 3);
      break;

    case 2:  // prismatic
      pjoint = mjs_addJoint(pbody, 0);
      mjs_setName(pjoint->element, jntname.c_str());
      pjoint->type = mjJNT_SLIDE;
      mjuu_setvec(pjoint->pos, 0, 0, 0);
      mjuu_copyvec(pjoint->axis, axis, 3);
      break;

    case 3:  // fixed- no joint, return
      return;

    case 4:  // floating
      pjoint = mjs_addJoint(pbody, 0);
      mjs_setName(pjoint->element, jntname.c_str());
      pjoint->type = mjJNT_FREE;
      break;

    case 5:  // planar- construct complex joint
      // make frame with axis = z
      mjuu_z2quat(quat, axis);
      mjuu_quat2mat(mat, quat);

      // construct slider along x
      pjoint = mjs_addJoint(pbody, 0);
      mjs_setName(pjoint->element, (jntname + "_TX").c_str());
      pjoint->type = mjJNT_SLIDE;
      tmpaxis[0]   = mat[0];
      tmpaxis[1]   = mat[3];
      tmpaxis[2]   = mat[6];
      mjuu_setvec(pjoint->pos, 0, 0, 0);
      mjuu_copyvec(pjoint->axis, tmpaxis, 3);

      // construct slider along y
      pjoint1 = mjs_addJoint(pbody, 0);
      mjs_setName(pjoint1->element, (jntname + "_TY").c_str());
      pjoint1->type = mjJNT_SLIDE;
      tmpaxis[0]    = mat[1];
      tmpaxis[1]    = mat[4];
      tmpaxis[2]    = mat[7];
      mjuu_setvec(pjoint1->pos, 0, 0, 0);
      mjuu_copyvec(pjoint1->axis, tmpaxis, 3);

      // construct hinge around z = locaxis
      pjoint2 = mjs_addJoint(pbody, 0);
      mjs_setName(pjoint2->element, (jntname + "_RZ").c_str());
      pjoint2->type = mjJNT_HINGE;
      mjuu_setvec(pjoint2->pos, 0, 0, 0);
      mjuu_copyvec(pjoint2->axis, axis, 3);
      break;

    case 6:  // ball joint
      pjoint = mjs_addJoint(pbody, 0);
      mjs_setName(pjoint->element, jntname.c_str());
      pjoint->type = mjJNT_BALL;
      mjuu_setvec(pjoint->pos, 0, 0, 0);
      mjuu_copyvec(pjoint->axis, axis, 3);
  }

  // dynamics element
  if ((elem = FindSubElem(joint_elem, "dynamics"))) {
    ReadAttr(elem, "damping", 1, pjoint->damping, text);
    ReadAttr(elem, "friction", 1, &pjoint->frictionloss, text);

    // copy parameters to all elements of planar joint
    if (pjoint1) {
      mjuu_copyvec(pjoint1->damping, pjoint->damping, 1 + mjNPOLY);
      mjuu_copyvec(pjoint2->damping, pjoint->damping, 1 + mjNPOLY);
      pjoint1->frictionloss = pjoint2->frictionloss = pjoint->frictionloss;
    }
  }

  // limit element
  if ((elem = FindSubElem(joint_elem, "limit"))) {
    if (jointtype == 1) {
      pjoint->limited = mjLIMITED_FALSE;
    } else {
      bool haslower = ReadAttr(elem, "lower", 1, pjoint->range, text);
      bool hasupper = ReadAttr(elem, "upper", 1, pjoint->range + 1, text);

      // handle range mis-specification, otherwise mjLIMITED_AUTO does the right thing
      bool bad_range = (haslower != hasupper) || pjoint->range[0] >= pjoint->range[1];
      if (haslower && hasupper && pjoint->range[0] == pjoint->range[1]) {
        mju_warning("joint '%s' has equal lower and upper limits, limit ignored", jntname.c_str());
      }
      if (bad_range) { pjoint->limited = mjLIMITED_FALSE; }
    }

    // ReadAttr(elem, "velocity", 1, &pjoint->maxvel, text); // no maxvel in MuJoCo
    double effort = 0;
    ReadAttr(elem, "effort", 1, &effort, text);
    effort = std::abs(effort);
    if (effort > 0) {
      pjoint->actfrcrange[0] = -effort;
      pjoint->actfrcrange[1] = effort;
    }
  }

  // mimic: q = multiplier*q_target + offset; deferred to a joint equality since
  // the target joint may be declared later
  if ((elem = FindSubElem(joint_elem, "mimic"))) {
    std::string target;
    ReadAttrTxt(elem, "joint", target, true);
    double multiplier = 1, offset = 0;
    ReadAttr(elem, "multiplier", 1, &multiplier, text);
    ReadAttr(elem, "offset", 1, &offset, text);
    urMimic.emplace_back(jntname, GetPrefixedName(target), multiplier, offset);
  }
}


// parse origin and geometry elements of visual or collision
mjsGeom* mjXURDF::Geom(XMLElement* geom_elem, mjsBody* pbody, bool collision) {
  XMLElement *elem, *temp;
  std::string text, meshfile;

  // get geometry element
  elem = FindSubElem(geom_elem, "geometry", true);

  // add BOX geom, modify type later
  mjsGeom* pgeom = mjs_addGeom(pbody, 0);
  mjs_setName(pgeom->element, "");
  pgeom->type = mjGEOM_BOX;
  if (collision) {
    pgeom->contype     = 1;
    pgeom->conaffinity = 1;
  } else {
    pgeom->contype     = 0;
    pgeom->conaffinity = 0;
    pgeom->group       = 1;
    pgeom->density     = 0;
  }

  // box
  if ((temp = FindSubElem(elem, "box"))) {
    ReadAttr(temp, "size", 3, pgeom->size, text, true, true);
    for (int i = 0; i < 3; i++) {
      pgeom->size[i] /= 2;  // MuJoCo uses half-length
    }
  }

  // cylinder
  else if ((temp = FindSubElem(elem, "cylinder"))) {
    pgeom->type = mjGEOM_CYLINDER;
    ReadAttr(temp, "radius", 1, pgeom->size, text, true, true);
    ReadAttr(temp, "length", 1, pgeom->size + 1, text, true, true);
    pgeom->size[1] /= 2;  // MuJoCo uses half-length
  }

  // sphere
  else if ((temp = FindSubElem(elem, "sphere"))) {
    pgeom->type = mjGEOM_SPHERE;
    ReadAttr(temp, "radius", 1, pgeom->size, text, true, true);
  }

  // capsule
  else if ((temp = FindSubElem(elem, "capsule"))) {
    pgeom->type = mjGEOM_CAPSULE;
    ReadAttr(temp, "radius", 1, pgeom->size, text, true, true);
    ReadAttr(temp, "length", 1, pgeom->size + 1, text, true, true);
    pgeom->size[1] /= 2;  // MuJoCo uses half-length
  }

  // mesh
  else if ((temp = FindSubElem(elem, "mesh"))) {
    mjsMesh* pmesh   = 0;
    bool     newmesh = false;

    // set geom type and read mesh attributes
    pgeom->type = mjGEOM_MESH;
    meshfile    = ReadAttrStr(temp, "filename", true).value();

    std::array<double, 3> default_meshscale = {1, 1, 1};
    std::array<double, 3> meshscale =
        ReadAttrArr<double, 3>(temp, "scale").value_or(default_meshscale);

    // strip file name if necessary
    if (spec->strippath) { meshfile = mjuu_strippath(meshfile); }

    // construct mesh name: always stripped
    std::string meshname = mjuu_strippath(meshfile);
    meshname             = mjuu_stripext(meshname);

    if (meshes.find(meshname) == meshes.end()) {
      // does not exist: create
      pmesh = mjs_addMesh(spec, 0);
      meshes[meshname].push_back(pmesh);
      newmesh = true;
    } else {
      int i = 0;

      // find if it exists with the same scale
      for (mjsMesh* mesh : meshes[meshname]) {
        if (*mesh->file == meshfile &&
            mesh->scale[0] == meshscale[0] &&
            mesh->scale[1] == meshscale[1] &&
            mesh->scale[2] == meshscale[2]) {
          pmesh = mesh;
          break;
        }
        i++;
      }

      // add a new spec making an incremental new name
      if (i == meshes[meshname].size()) {
        pmesh = mjs_addMesh(spec, 0);
        meshes[meshname].push_back(pmesh);
        newmesh = true;
      }

      // if it is not the first mesh with this name, append index
      if (i > 0) { meshname = meshname + std::to_string(i); }
    }

    // set fields
    if (newmesh) {
      // make the name unique, an indexed name can clash with another mesh
      std::string uniquename = meshname;
      for (int k = 1; mjs_findElement(spec, mjOBJ_MESH, uniquename.c_str()); k++) {
        uniquename = meshname + "_" + std::to_string(k);
      }
      mjs_setString(pmesh->file, meshfile.c_str());
      mjs_setName(pmesh->element, uniquename.c_str());
      pmesh->scale[0] = meshscale[0];
      pmesh->scale[1] = meshscale[1];
      pmesh->scale[2] = meshscale[2];
    }
    mjs_setString(pgeom->meshname, mjs_getString(mjs_getName(pmesh->element)));
  }

  else {
    throw mjXError(elem, "visual geometry specification not found");
  }

  // origin element
  Origin(geom_elem, pgeom->pos, pgeom->quat);

  return pgeom;
}


// parse origin element
void mjXURDF::Origin(XMLElement* origin_elem, double* pos, double* quat) {
  XMLElement* temp;
  std::string text;

  // set defaults
  mjuu_setvec(pos, 0, 0, 0);
  mjuu_setvec(quat, 1, 0, 0, 0);

  // read origin element if present
  if ((temp = FindSubElem(origin_elem, "origin"))) {
    // position
    ReadAttr(temp, "xyz", 3, pos, text);

    // orientation
    mjsOrientation alt;
    mjs_defaultOrientation(&alt);
    if (ReadAttr(temp, "rpy", 3, alt.euler, text)) {
      alt.type = mjORIENTATION_EULER;
      mjs_resolveOrientation(quat, 0, "XYZ", &alt);
    }
  }
}


// find body with given name in list, return -1 if not found
int mjXURDF::FindName(std::string name, std::vector<std::string>& list) {
  for (unsigned int i = 0; i < list.size(); i++)
    if (list[i] == name) { return i; }

  return -1;
}


// add name to list, error if name already exists
void mjXURDF::AddName(std::string name, std::vector<std::string>& list) {
  // make sure name is unique
  if (FindName(name, list) >= 0) { throw mjXError(0, "repeated URDF name: ", name.c_str()); }

  list.push_back(name);
}


// add body name to list of URDF bodies, error if name already exists
void mjXURDF::AddBody(std::string name) {
  // add body name, make sure it is unique
  AddName(name, urName);

  // add parent and child elements
  urParent.push_back(-1);
  std::vector<int> children;
  children.clear();
  urChildren.push_back(children);
}


// add body with given number to the mjCModel tree, process children
void mjXURDF::AddToTree(int n) {
  // get pointer to parent in mjCModel tree
  mjsBody *parent = 0, *child = 0, *world = 0;
  if (urParent[n] >= 0) {
    world  = mjs_findBody(spec, "world");
    parent = mjs_findChild(world, urName[urParent[n]].c_str());

    if (!parent)
      throw mjXError(0,
                     "URDF body parent should already be in tree: %s",
                     urName[urParent[n]].c_str());  // SHOULD NOT OCCUR
  } else {
    parent = mjs_findBody(spec, "world");
  }

  // add this body
  if (urName[n] != "world") {
    child = mjs_addBody(parent, 0);
    mjs_setName(child->element, urName[n].c_str());
  }

  // add children recursively
  for (int i = 0; i < (int)urChildren[n].size(); i++) { AddToTree(urChildren[n][i]); }
}


// find all materials recursively
void mjXURDF::MakeMaterials(XMLElement* elem) {
  std::string name, text;
  XMLElement* color = 0;
  mjRGBA      rgba;

  // process this element
  if (!std::strcmp(elem->Value(), "material")) {
    // make sure material is named
    if (ReadAttrTxt(elem, "name", name)) {
      // make sure name is not already registered
      if (FindName(name, urMat) < 0) {
        // add rgba value if available
        if ((color = FindSubElem(elem, "color"))) {
          ReadAttr(color, "rgba", 4, rgba.val, text, /*required=*/true);
          AddName(name, urMat);
          urRGBA.push_back(rgba);
        }
      }
    }
  }

  // process children recursively
  elem = elem->FirstChildElement();
  while (elem) {
    MakeMaterials(elem);
    elem = elem->NextSiblingElement();
  }
}
