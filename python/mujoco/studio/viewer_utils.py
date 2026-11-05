# Copyright 2026 DeepMind Technologies Limited
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Utility functions for Studio viewer implementations."""

import mujoco
from mujoco.studio import messages
from mujoco.studio import viewer_protocol
import numpy as np


def send_state(
    viewer: viewer_protocol.Viewer,
    model: mujoco.MjModel,
    data: mujoco.MjData,
    state_sig: int,
) -> None:
  """Sends the selected components of the viewer's state to the sim thread."""
  event = messages.StateEvent.from_data(model, data, state_sig)
  if event is not None:
    viewer.send_to_sim(event)


def apply_perturb(
    viewer: viewer_protocol.Viewer,
    model: mujoco.MjModel | None,
    data: mujoco.MjData | None,
    is_paused: bool = False,
) -> None:
  """Apply perturbation to the model and send updated forces to the sim thread."""
  if model is None or data is None:
    return
  perturb = viewer.perturb
  if not is_paused:
    sig = int(mujoco.mjtState.mjSTATE_XFRC_APPLIED)
    size = mujoco.mj_stateSize(model, sig)
    zero_state = np.zeros(size, np.float64)
    mujoco.mj_setState(model, data, zero_state, sig)
    # Running: the pose edit moves a dragged mocap body in the viewer's local
    # data; send it to the sim, else the next StateSnapshot reverts it.
    mocap_sig = (
        mujoco.mjtState.mjSTATE_MOCAP_POS.value
        | mujoco.mjtState.mjSTATE_MOCAP_QUAT.value
    )
    with viewer.send_if_changes(
        lambda: messages.StateEvent.from_data(model, data, mocap_sig)
    ):
      mujoco.mjv_applyPerturbPose(model, data, perturb, 0)
    mujoco.mjv_applyPerturbForce(model, data, perturb)
  else:
    # Paused: the pose edit only moves the viewer's local data; send it to the
    # sim, else the next StateSnapshot reverts it when the drag ends.
    pose_sig = (
        mujoco.mjtState.mjSTATE_QPOS.value
        | mujoco.mjtState.mjSTATE_MOCAP_POS.value
        | mujoco.mjtState.mjSTATE_MOCAP_QUAT.value
    )
    with viewer.send_if_changes(
        lambda: messages.StateEvent.from_data(model, data, pose_sig)
    ):
      mujoco.mjv_applyPerturbPose(model, data, perturb, 1)

  xfrc_sig: int = int(mujoco.mjtState.mjSTATE_XFRC_APPLIED)
  xfrc_size: int = mujoco.mj_stateSize(model, xfrc_sig)
  xfrc_state: np.ndarray = np.zeros(xfrc_size, np.float64)

  mujoco.mj_getState(model, data, xfrc_state, xfrc_sig)
  viewer.send_to_sim(
      messages.PerturbSnapshot(state=xfrc_state, state_sig=xfrc_sig)
  )
