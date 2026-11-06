# Copyright 2025 DeepMind Technologies Limited
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

# Find OpenUSD package. The user can either provide pxr_DIR to use their own
# OpenUSD installation or they can build external_dependencies.
#
# We use QUIET here to provide a better error message when pxr is not found.
find_package(pxr
  QUIET
  HINTS "${CMAKE_BINARY_DIR}/_deps/openusd-build"
)

if(NOT pxr_FOUND)
    message(FATAL_ERROR
        "-----------------------------------------------------------\n"
        "USD Configuration Error:\n${pxr_NOT_FOUND_MESSAGE}\n"
        "If you have built USD yourself, provide -Dpxr_DIR=your_pxr_install_dir' \n"
        "Otherwise you can build cmake/third_party_deps/openusd:\n"
        "cd ~/mujoco\n"
        "cmake -Bcmake/third_party_deps/openusd/build cmake/third_party_deps/openusd -DBUILD_USD=True\n"
        "cmake --build cmake/third_party_deps/openusd/build\n"
        "cmake -Bbuild -S. -DMUJOCO_WITH_USD=True\n"
        "cmake --build build\n"
        "cmake --install build"
        "-----------------------------------------------------------"
    )
endif()

if(WIN32)
  add_compile_definitions(NOMINMAX)

  # Link pure-C++ USD libraries against CPython's version-independent
  # python3.lib forwarder (PEP 384) and suppress pyconfig.h's auto-linked
  # python3XX.lib so bundled USD DLLs load under any Python 3.11+ interpreter.
  set(_mj_py_lib_dirs ${Python3_LIBRARY_DIRS})
  if(TARGET Python3::Python)
    foreach(_mj_py_prop IMPORTED_IMPLIB_RELEASE IMPORTED_IMPLIB)
      get_target_property(_mj_py_implib Python3::Python ${_mj_py_prop})
      if(_mj_py_implib)
        get_filename_component(_mj_py_dir "${_mj_py_implib}" DIRECTORY)
        list(APPEND _mj_py_lib_dirs "${_mj_py_dir}")
      endif()
    endforeach()
  endif()
  foreach(_mj_py_cand ${Python3_LIBRARY_RELEASE} ${Python3_LIBRARY} ${Python3_LIBRARIES})
    if(EXISTS "${_mj_py_cand}" AND NOT IS_DIRECTORY "${_mj_py_cand}")
      get_filename_component(_mj_py_dir "${_mj_py_cand}" DIRECTORY)
      list(APPEND _mj_py_lib_dirs "${_mj_py_dir}")
    endif()
  endforeach()
  list(REMOVE_DUPLICATES _mj_py_lib_dirs)

  set(_mj_python3_stable_lib "")
  foreach(_mj_py_dir ${_mj_py_lib_dirs})
    if(EXISTS "${_mj_py_dir}/python3.lib")
      set(_mj_python3_stable_lib "${_mj_py_dir}/python3.lib")
      break()
    endif()
  endforeach()

  if(_mj_python3_stable_lib)
    set(_mj_pxr_targets ${PXR_LIBRARIES} python boost pxr_boost_python arch)
    if(TARGET usd_m)
      get_target_property(_mj_usd_m_deps usd_m INTERFACE_LINK_LIBRARIES)
      if(_mj_usd_m_deps)
        list(APPEND _mj_pxr_targets ${_mj_usd_m_deps})
      endif()
    endif()
    list(REMOVE_DUPLICATES _mj_pxr_targets)

    foreach(_mj_tgt ${_mj_pxr_targets})
      if(TARGET ${_mj_tgt})
        get_target_property(_mj_libs ${_mj_tgt} INTERFACE_LINK_LIBRARIES)
        if(_mj_libs)
          list(TRANSFORM _mj_libs REPLACE "^Python3::.*$" "${_mj_python3_stable_lib}")
          set_target_properties(${_mj_tgt} PROPERTIES INTERFACE_LINK_LIBRARIES "${_mj_libs}")
        endif()
      endif()
    endforeach()

    if(Python3_VERSION_MAJOR AND Python3_VERSION_MINOR)
      set(_mj_py_ver "${Python3_VERSION_MAJOR}${Python3_VERSION_MINOR}")
      set(_mj_nodefault_tgt "")
      if(TARGET arch)
        set(_mj_nodefault_tgt arch)
      elseif(TARGET usd_m)
        set(_mj_nodefault_tgt usd_m)
      endif()
      if(_mj_nodefault_tgt)
        set_property(
          TARGET ${_mj_nodefault_tgt}
          APPEND
          PROPERTY
            INTERFACE_LINK_OPTIONS
            "LINKER:/NODEFAULTLIB:python${_mj_py_ver}.lib"
            "LINKER:/NODEFAULTLIB:python${_mj_py_ver}_d.lib"
        )
      endif()
    endif()
  endif()
endif()
