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

#include "kinematic_tree.h"
#include "newton_tokens.h"

#include <map>
#include <memory>
#include <utility>
#include <vector>

#include <mujoco/experimental/usd/mjcPhysics/actuator.h>
#include <mujoco/experimental/usd/mjcPhysics/equalityJointAPI.h>
#include <mujoco/experimental/usd/mjcPhysics/keyframe.h>
#include <mujoco/experimental/usd/mjcPhysics/siteAPI.h>
#include <mujoco/experimental/usd/mjcPhysics/tendon.h>
#include <mujoco/mujoco.h>
#include <pxr/usd/sdf/path.h>
#include <pxr/usd/usd/common.h>
#include <pxr/usd/usd/primFlags.h>
#include <pxr/usd/usd/primRange.h>
#include <pxr/usd/usdGeom/gprim.h>
#include <pxr/usd/usdGeom/xformCache.h>
#include <pxr/usd/usdPhysics/articulationRootAPI.h>
#include <pxr/usd/usdPhysics/collisionAPI.h>
#include <pxr/usd/usdPhysics/joint.h>
#include <pxr/usd/usdPhysics/rigidBodyAPI.h>
#include <pxr/usd/usdPhysics/scene.h>

using mujoco::NewtonTokens;

bool GetJointBodies(const pxr::UsdPhysicsJoint& joint, pxr::SdfPath* from,
                    pxr::SdfPath* to) {
  // Grab the default prim path
  pxr::SdfPath default_prim_path;
  auto stage = joint.GetPrim().GetStage();
  if (stage->GetDefaultPrim().IsValid()) {
    default_prim_path = stage->GetDefaultPrim().GetPath();
  }

  pxr::SdfPathVector body1_paths;
  joint.GetBody1Rel().GetTargets(&body1_paths);
  if (body1_paths.empty()) {
    mju_warning("Joint %s does not have body1 rel. Skipping.",
                joint.GetPath().GetAsString().c_str());
    return false;
  } else if (body1_paths.size() > 1) {
    mju_warning("Joint %s has multiple body1 rels. Skipping.",
                joint.GetPath().GetAsString().c_str());
    return false;
  }
  // body1 pointing to the default prim means it's attached to the worldbody.
  if (body1_paths[0] == default_prim_path) {
    *to = pxr::SdfPath();
  } else {
    *to = body1_paths[0];
  }

  pxr::SdfPathVector body0_paths;
  joint.GetBody0Rel().GetTargets(&body0_paths);
  if (body0_paths.size() > 1) {
    mju_warning("Joint %s has multiple body0 rels. Skipping.",
                joint.GetPath().GetAsString().c_str());
    return false;
  }
  // Empty body0, or body0 pointing to the default prim means we'll attach
  // to the worldbody.
  if (body0_paths.empty() || body0_paths[0] == default_prim_path) {
    *from = pxr::SdfPath();
  } else {
    *from = body0_paths[0];
  }
  return true;
}

struct ExtractedPrims {
  std::vector<std::unique_ptr<Node>> nodes;
  std::vector<pxr::UsdPhysicsJoint> joints;
};

ExtractedPrims ExtractPrims(pxr::UsdStageRefPtr stage) {
  pxr::UsdPhysicsScene physics_scene;
  std::vector<std::unique_ptr<Node>> nodes;
  std::vector<pxr::UsdPhysicsJoint> joints;
  nodes.push_back(std::make_unique<Node>());
  Node* root = nodes.back().get();

  // =========================================================================
  // PASS 1: Collect Bodies, Joints, and Geoms/Sites/etc.
  // =========================================================================
  // A single DFS pass to find all bodies, joints, and determine
  // which body owns each geom/site/etc. prim.
  std::vector<Node*> owner_stack;
  owner_stack.push_back(root);  // Start with the world as owner.

  const auto range = pxr::UsdPrimRange::PreAndPostVisit(
      stage->GetPseudoRoot(), pxr::UsdTraverseInstanceProxies());

  pxr::UsdGeomXformCache xform_cache;
  for (auto it = range.begin(); it != range.end(); ++it) {
    pxr::UsdPrim prim = *it;

    bool is_body = prim.HasAPI<pxr::UsdPhysicsRigidBodyAPI>() ||
                   prim.HasAPI<pxr::UsdPhysicsArticulationRootAPI>();
    bool resets = xform_cache.GetResetXformStack(prim);
    // Only update (push/pop) the owner stack for bodies (becomes new owner) and
    // resetXformStack (reset owner to world).
    bool is_pushed_to_stack = is_body || resets;

    if (it.IsPostVisit()) {
      if (is_pushed_to_stack) {
        owner_stack.pop_back();
      }
      continue;
    }

    pxr::SdfPath prim_path = prim.GetPath();
    Node* current_node = owner_stack.back();

    if (is_body) {
      auto new_node = std::make_unique<Node>();
      new_node->body_path = prim_path;
      nodes.push_back(std::move(new_node));
      current_node = nodes.back().get();
    } else if (resets) {
      current_node = root;  // Reset owner to world.
    }

    if (is_pushed_to_stack) {
      owner_stack.push_back(current_node);
    }

    if (prim.IsA<pxr::UsdPhysicsScene>() && root->physics_scene.IsEmpty()) {
      root->physics_scene = prim_path;
    }

    if (prim.HasAPI(NewtonTokens->NewtonSiteAPI) ||
        prim.HasAPI<pxr::MjcPhysicsSiteAPI>()) {
      current_node->sites.push_back(prim_path);
      // Sites should not have children.
      it.PruneChildren();
    } else if (prim.IsA<pxr::UsdGeomGprim>()) {
      bool has_collision_api = prim.HasAPI<pxr::UsdPhysicsCollisionAPI>();
      if (has_collision_api) {
        bool collision_enabled = false;
        pxr::UsdPhysicsCollisionAPI(prim).GetCollisionEnabledAttr().Get(
            &collision_enabled);
        if (collision_enabled) {
          current_node->colliders.push_back(prim_path);
        } else {
          current_node->visual_gprims.push_back(prim_path);
        }
      } else {
        current_node->visual_gprims.push_back(prim_path);
      }
    }

    if (prim.IsA<pxr::UsdPhysicsJoint>()) {
      // We may not know which body this belongs to yet so we'll add it to a
      // list and the caller can assign the joints when building the tree.
      joints.push_back(pxr::UsdPhysicsJoint(prim));
      // Joints should not have children.
      it.PruneChildren();
    }

    if (prim.IsA<pxr::MjcPhysicsActuator>()) {
      root->actuators.push_back(prim_path);
      // Joints should not have children.
      it.PruneChildren();
    }

    if (prim.IsA<pxr::MjcPhysicsTendon>()) {
      root->tendons.push_back(prim_path);
      // Tendons should not have children of interest.
      it.PruneChildren();
    }

    if (prim.IsA<pxr::MjcPhysicsKeyframe>()) {
      root->keyframes.push_back(prim_path);
      // Keyframes should not have children.
      it.PruneChildren();
    }
  }
  return {std::move(nodes), std::move(joints)};
}

std::unique_ptr<Node> BuildKinematicTree(const pxr::UsdStageRefPtr stage) {
  ExtractedPrims extraction = ExtractPrims(stage);

  std::map<pxr::SdfPath, int> body_index;
  body_index[pxr::SdfPath()] = 0;
  for (int i = 0; i < extraction.nodes.size(); ++i) {
    body_index[extraction.nodes[i]->body_path] = i;
  }

  // List of direct children for each body.
  std::vector<std::vector<bool>> children(
      extraction.nodes.size(), std::vector<bool>(extraction.nodes.size()));
  // List of joint prim paths associated with a child for each body.
  std::vector<std::vector<pxr::SdfPath>> parent_joints(extraction.nodes.size());
  for (const pxr::UsdPhysicsJoint& joint : extraction.joints) {
    pxr::SdfPath from, to;
    if (!GetJointBodies(joint, &from, &to)) {
      continue;
    }

    // Check if this is a constraint (not participating in kinematic tree).
    bool excluded_from_articulation = false;
    joint.GetExcludeFromArticulationAttr().Get(&excluded_from_articulation);

    // Check if bodies exist in body_index (don't use [] which inserts defaults)
    auto from_it = body_index.find(from);
    auto to_it = body_index.find(to);

    if (from_it == body_index.end() || to_it == body_index.end()) {
      // For constraint joints (e.g., site-based welds), body0/body1 may point
      // to sites which aren't in body_index. Add these to the world node.
      if (excluded_from_articulation) {
        extraction.nodes[0]->constraints.push_back(joint.GetPath());
      }
      continue;
    }
    int from_idx = from_it->second;
    int to_idx = to_it->second;
    if (from_idx == to_idx) {
      mju_error("Cycle detected: self referencing joint at node %s",
                to.GetString().c_str());
      return nullptr;
    }

    // If we encounter a joint that does not participate in articulation, we
    // should treat it as a constraint instead.
    // For example, a weld constraint is represented by a fixed joint.
    if (excluded_from_articulation) {
      extraction.nodes[to_idx]->constraints.push_back(joint.GetPath());
      continue;
    }

    children[from_idx][to_idx] = true;
    parent_joints[to_idx].push_back(joint.GetPath());
    // Now that we know all the bodies, we can assign joints to respective
    // nodes.
    extraction.nodes[to_idx]->joints.push_back(joint.GetPath());

    // If the joint has NewtonMimicAPI or MjcPhysicsEqualityJointAPI, also add
    // it to constraints so that ParseConstraint is called to create the
    // equality constraint.
    if (joint.GetPrim().HasAPI(NewtonTokens->NewtonMimicAPI) ||
        joint.GetPrim().HasAPI<pxr::MjcPhysicsEqualityJointAPI>()) {
      extraction.nodes[to_idx]->constraints.push_back(joint.GetPath());
    }
  }

  // The world body is represented by an empty SdfPath.
  auto world_root = std::move(extraction.nodes[0]);

  std::vector<std::pair<int, Node*>> stack;
  stack.emplace_back(0, world_root.get());

  // A node without any joints from a parent has a free joint.
  // Add all free joints as children of the world body.
  for (int i = 1; i < extraction.nodes.size(); ++i) {
    if (parent_joints[i].empty()) {
      children[0][i] = true;
    }
  }

  std::vector<bool> visited(extraction.nodes.size());
  visited[0] = true;
  while (!stack.empty()) {
    auto [current_body, parent] = stack.back();
    stack.pop_back();

    Node* current_node = nullptr;
    if (current_body > 0) {
      parent->children.push_back(std::move(extraction.nodes[current_body]));
      current_node = parent->children.back().get();
    } else {
      current_node = world_root.get();
    }

    // Process children in reverse to maintain DFS.
    for (int i = extraction.nodes.size() - 1; i > 0; --i) {
      if (!children[current_body][i]) {
        continue;
      }
      if (visited[i]) {
        mju_error("Cycle detected in the kinematic tree at node %s",
                  current_node->body_path.GetString().c_str());
        return nullptr;
      }
      visited[i] = true;
      stack.emplace_back(
          i, current_body > 0 ? parent->children.back().get() : parent);
    }
  }

  for (int i = 1; i < visited.size(); ++i) {
    if (!visited[i]) {
      mju_error("Cycle detected: Node %s is not reachable from the world.",
                extraction.nodes[i]->body_path.GetString().c_str());
      return nullptr;
    }
  }
  return world_root;
}
