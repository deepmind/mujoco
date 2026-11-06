Building
========

.. WARNING:: OpenUSD support is currently experimental and subject to frequent change.

.. tip::
   If you are using Python 3.11--3.13 on Linux ``x86_64``, Windows ``x86_64``, or macOS ``arm64``, you can install
   prebuilt C++ OpenUSD bindings directly from PyPI with ``pip install mujoco-usd`` (or ``pip install mujoco[usd]``)
   without building OpenUSD or MuJoCo from source.

MuJoCo must be built against a pre-built USD library, we provide a utility to do so but you may also bring your own USD
libraries.

The following instructions assume that you have cloned MuJoCo into ``~/mujoco`` and have a build directory at
``~/mujoco/build``.

.. _usdBuildingUSD:

Building USD
------------

If you have a pre-built USD library, you can skip this section.

MuJoCo provides a CMake project that simplifies the process of building USD. It will download and build USD with only
the necessary features enabled.

.. code-block:: bash

   cd ~/mujoco
   cmake -Bcmake/third_party_deps/openusd/build cmake/third_party_deps/openusd
   cmake --build cmake/third_party_deps/openusd/build

If you want to customize the build process, you can use USD's ``build_usd.py`` script. It's recommended to use a
separate installation directory that exists outside of the cloned repository directory.

.. code-block:: bash

   git clone https://github.com/PixarAnimationStudios/OpenUSD
   python OpenUSD/build_scripts/build_usd.py /path/to/my_usd_install_dir

.. _usdEnablingUSD:

Enabling USD
------------

If USD was built with the third_party_deps/openusd CMake project, you can enable USD support with the MUJOCO_WITH_USD
flag.

.. code-block:: bash

   cd ~/mujoco
   cmake -Bbuild -S. -DMUJOCO_WITH_USD=True
   cmake --build build -j 64

Otherwise, if you have a pre-built USD library, you must also pass the pxr_DIR flag.

.. code-block:: bash

   cd ~/mujoco
   cmake -Bbuild -S. -DMUJOCO_WITH_USD=True -Dpxr_DIR=/path/to/my_usd_install_dir
   cmake --build build -j 64


If we now run :ref:`simulate.cc <saSimulate>`, we will be able to drag and drop USD files.

.. code-block:: bash

   simulate
