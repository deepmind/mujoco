# Copyright 2026 DeepMind Technologies Limited
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Tests for EGL device selection."""

import contextlib
import importlib.util
import os
import pathlib
import sys
import types
from unittest import mock

from absl.testing import absltest


class _FakeGLError(Exception):
  pass


class _FakeEGL:
  # pylint: disable=invalid-name

  EGL_TRUE = 1
  EGL_SUCCESS = 0x3000
  EGL_NO_DISPLAY = None
  EGL_PLATFORM_DEVICE_EXT = 0x313F
  EGL_CUDA_DEVICE_NV = 0x323A

  EGL_RED_SIZE = 0
  EGL_GREEN_SIZE = 1
  EGL_BLUE_SIZE = 2
  EGL_ALPHA_SIZE = 3
  EGL_DEPTH_SIZE = 4
  EGL_STENCIL_SIZE = 5
  EGL_COLOR_BUFFER_TYPE = 6
  EGL_RGB_BUFFER = 7
  EGL_SURFACE_TYPE = 8
  EGL_PBUFFER_BIT = 9
  EGL_RENDERABLE_TYPE = 10
  EGL_OPENGL_BIT = 11
  EGL_NONE = 12

  def __init__(self, devices, cuda_ids=None):
    self._devices = devices
    self._cuda_ids = cuda_ids

  def eglQueryDevicesEXT(self):
    return self._devices

  def eglQueryDeviceAttribEXT(self, device, attribute):
    if self._cuda_ids is None:
      raise ImportError('eglQueryDeviceAttribEXT is not available.')
    if attribute != self.EGL_CUDA_DEVICE_NV or device not in self._cuda_ids:
      raise _FakeGLError()
    return self._cuda_ids[device]

  def eglGetPlatformDisplayEXT(self, platform, device, attributes):
    del platform, attributes
    return f'display:{device}'

  def eglGetError(self):
    return self.EGL_SUCCESS

  def eglInitialize(self, display, major, minor):
    del display, major, minor
    return self.EGL_TRUE


def _load_egl_module(fake_egl, env):
  fake_mujoco = types.ModuleType('mujoco')
  fake_mujoco.__path__ = []
  fake_mujoco_egl = types.ModuleType('mujoco.egl')
  fake_mujoco_egl.__path__ = []
  fake_mujoco_egl.egl_ext = fake_egl
  fake_mujoco_google = types.ModuleType('mujoco.google')
  fake_mujoco_google._pyopengl_patch = types.SimpleNamespace(
      patch_ctypes_cdll=contextlib.nullcontext
  )
  fake_mujoco.google = fake_mujoco_google
  fake_opengl = types.ModuleType('OpenGL')
  fake_opengl.error = types.SimpleNamespace(GLError=_FakeGLError)

  module_name = '_mujoco_egl_under_test'
  module_path = pathlib.Path(__file__).with_name('egl') / '__init__.py'
  spec = importlib.util.spec_from_file_location(module_name, module_path)
  module = importlib.util.module_from_spec(spec)

  modules = {
      'mujoco': fake_mujoco,
      'mujoco.egl': fake_mujoco_egl,
      'mujoco.egl.egl_ext': fake_egl,
      'mujoco.google': fake_mujoco_google,
      'mujoco.google._pyopengl_patch': fake_mujoco_google._pyopengl_patch,
      'OpenGL': fake_opengl,
      'OpenGL.error': fake_opengl.error,
  }
  test_env = {'PYOPENGL_PLATFORM': 'egl'}
  test_env.update(env)

  with mock.patch.dict(sys.modules, modules), mock.patch.dict(
      os.environ, test_env, clear=True):
    spec.loader.exec_module(module)
  return module


def _create_initialized_egl_device_display(module, env):
  test_env = {'PYOPENGL_PLATFORM': 'egl'}
  test_env.update(env)
  with mock.patch.dict(os.environ, test_env, clear=True):
    return module.create_initialized_egl_device_display()


class EGLDeviceSelectionTest(absltest.TestCase):

  def test_selected_device_uses_cuda_device_id(self):
    devices = ['egl-for-cuda-2', 'egl-for-cuda-1', 'egl-for-cuda-0', 'egl-mesa']
    fake_egl = _FakeEGL(
        devices,
        {
            'egl-for-cuda-0': 0,
            'egl-for-cuda-1': 1,
            'egl-for-cuda-2': 2,
        })
    module = _load_egl_module(fake_egl, {'MUJOCO_EGL_DEVICE_ID': '2'})

    display = _create_initialized_egl_device_display(
        module, {'MUJOCO_EGL_DEVICE_ID': '2'})

    self.assertEqual(display, 'display:egl-for-cuda-2')

  def test_selected_device_uses_cuda_visible_logical_id(self):
    devices = ['egl-for-cuda-2', 'egl-for-cuda-0']
    fake_egl = _FakeEGL(
        devices,
        {
            'egl-for-cuda-0': 0,
            'egl-for-cuda-2': 2,
        })
    module = _load_egl_module(fake_egl, {
        'CUDA_VISIBLE_DEVICES': '2',
        'MUJOCO_EGL_DEVICE_ID': '0',
    })

    display = _create_initialized_egl_device_display(module, {
        'CUDA_VISIBLE_DEVICES': '2',
        'MUJOCO_EGL_DEVICE_ID': '0',
    })

    self.assertEqual(display, 'display:egl-for-cuda-2')

  def test_selected_device_disambiguates_logical_and_physical_ids(self):
    devices = ['egl-for-cuda-1', 'egl-for-cuda-2', 'egl-hidden']
    fake_egl = _FakeEGL(
        devices,
        {
            'egl-for-cuda-1': 0,
            'egl-for-cuda-2': 1,
        })
    env = {
        'CUDA_VISIBLE_DEVICES': '1,2',
        'MUJOCO_EGL_DEVICE_ID': '0',
    }
    module = _load_egl_module(fake_egl, env)

    display = _create_initialized_egl_device_display(module, env)

    self.assertEqual(display, 'display:egl-for-cuda-1')

  def test_selected_device_uses_cuda_visible_physical_id(self):
    devices = ['egl-hidden', 'egl-visible', 'egl-mesa']
    fake_egl = _FakeEGL(devices, {'egl-visible': 0})
    module = _load_egl_module(fake_egl, {
        'CUDA_VISIBLE_DEVICES': '2',
        'MUJOCO_EGL_DEVICE_ID': '2',
    })

    display = _create_initialized_egl_device_display(module, {
        'CUDA_VISIBLE_DEVICES': '2',
        'MUJOCO_EGL_DEVICE_ID': '2',
    })

    self.assertEqual(display, 'display:egl-visible')

  def test_falls_back_to_egl_order_without_cuda_query(self):
    devices = ['egl-0', 'egl-1', 'egl-2']
    fake_egl = _FakeEGL(devices)
    module = _load_egl_module(fake_egl, {'MUJOCO_EGL_DEVICE_ID': '1'})

    display = _create_initialized_egl_device_display(
        module, {'MUJOCO_EGL_DEVICE_ID': '1'})

    self.assertEqual(display, 'display:egl-1')

  def test_unselected_devices_use_cuda_visible_order(self):
    devices = ['egl-for-cuda-4', 'egl-for-cuda-2', 'egl-for-cuda-0']
    fake_egl = _FakeEGL(
        devices,
        {
            'egl-for-cuda-0': 0,
            'egl-for-cuda-2': 2,
            'egl-for-cuda-4': 4,
        })
    module = _load_egl_module(fake_egl, {'CUDA_VISIBLE_DEVICES': '2,4'})

    display = _create_initialized_egl_device_display(
        module, {'CUDA_VISIBLE_DEVICES': '2,4'})

    self.assertEqual(display, 'display:egl-for-cuda-2')


if __name__ == '__main__':
  absltest.main()
