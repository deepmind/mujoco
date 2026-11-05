Model Editing
-------------

It is possible to create and modify models using the :ref:`mjSpec` struct and related API.
This data structure is in one-to-one correspondence with MJCF and indeed, MuJoCo's own XML parsers (both MJCF and URDF)
use this API when loading a model.


.. _meOverview:

Overview
~~~~~~~~

The API augments the traditional workflow of creating and editing models using XML files, breaking up the *parse* and
*compile* steps. As summarized in the :ref:`Overview chapter<Instance>`, the traditional workflow is:

 1. Create an XML model description file (MJCF or URDF) and associated assets. |br|
 2. Call :ref:`mj_loadXML`, obtain an :ref:`mjModel` instance.

The workflow using :ref:`mjSpec` is:

 1. Create an empty :ref:`mjSpec` using :ref:`mj_makeSpec` or parse an existing XML file using :ref:`mj_parseXML`.
 2. Programmatically edit the :ref:`mjSpec` data structure by adding, modifying, and removing elements.
 3. Compile the :ref:`mjSpec` to an :ref:`mjModel` instance using :ref:`mj_compile`.

 After compilation, the :ref:`mjSpec` remains editable, so steps 2 and 3 are interchangeable.


.. _Load:
.. _meLoading:

Model Parsing & Loading
~~~~~~~~~~~~~~~~~~~~~~~

As summarized in :ref:`Model instances <Instance>`, model description files (MJCF, MJZ, URDF, USD) are parsed into an
:ref:`mjSpec` using :ref:`mj_parse` (or ``mjSpec.from_file()`` / ``mjSpec.from_string()`` in Python). The model format is
inferred from the content type or file extension, and parsing into an :ref:`mjSpec` is delegated to the appropriate
:ref:`decoder <exDecoder>` plugin.

.. code-block:: C

   char error[1000] = "";
   mjSpec* spec = mj_parse(vfs, "robot.xml", NULL, NULL, error, sizeof(error));

For convenience, :ref:`mj_loadXML` (or Python ``MjModel.from_xml_path()``) combines parsing and compilation into a
single step, returning a compiled :ref:`mjModel` directly from an XML file or ``.mjz`` archive.

Alternatively, a pre-compiled :ref:`mjModel` can be loaded directly from a binary MJB file using :ref:`mj_loadModel`
(or Python ``MjModel.from_binary_path()``).

.. _Compile:
.. _meCompilation:

Model Compilation
~~~~~~~~~~~~~~~~~

Once a high-level :ref:`mjSpec` is created---by parsing a file, loading an archive, or constructing it
programmatically---it is compiled into an :ref:`mjModel` using :ref:`mj_compile`.

Compilation is independent of loading, working in the exact same way regardless of how :ref:`mjSpec` was constructed.
Both the parser and the compiler perform extensive error checking and abort when the first error is encountered. The
parser uses a custom schema to validate file structure, elements, and attributes, while the compiler applies semantic
checks and executes a test simulation step to catch runtime errors.

Parsing and compilation are extremely fast—typically less than a second—making interactive model design, live editing,
and rapid reloading seamless.

.. _Save:
.. _meSaving:

Model Encoding & Saving
~~~~~~~~~~~~~~~~~~~~~~~
Model specs and compiled models can be serialized to files using :ref:`mj_encode`, or directly saved to XML strings
using :ref:`mj_saveXMLString` or :ref:`mj_saveXML`.

The :ref:`mj_encode` function provides a unified entry point for serializing models:

.. code-block:: C

   char error[1024] = "";
   mjtSize bytes_written = mj_encode(spec, model, "robot.mjz", NULL, vfs, error, sizeof(error));

The output format is selected automatically based on the file extension (case-insensitive) or explicit ``content_type``:

- **MJCF XML** (``.xml``): Flattens the spec into a single MJCF XML file using :ref:`mj_saveXML`. If an explicit
  :ref:`mjModel` argument is passed, :ref:`mj_encode` will copy modified values back from ``mjModel`` into the spec
  prior to saving. In the Computation chapter we show an `example <_static/example.xml>`__ MJCF file and the
  corresponding `saved example <_static/example_saved.xml>`__.
- **MJZ Archive** (``.mjz`` or ``.zip``): Bundles the spec and all associated external assets (meshes, textures,
  included XMLs) into a self-contained Zip archive via the built-in ``mjz_encoder``.
- **MJB Binary** (``.mjb``): Serializes the compiled :ref:`mjModel` in MuJoCo binary format via :ref:`mj_saveModel`.
  MJB files are standalone, do not refer to external files, and load faster than XML, but are version-specific and
  cannot be decompiled back to XML. Requires a compiled ``model``; does **not** serialize anything from ``spec``.
- **TXT** (``.txt``): Writes a human-readable text dump via :ref:`mj_printModel`. Useful for diffing and debugging.
  Requires a compiled ``model``; does **not** serialize anything from ``spec``.

A spec is saved as MJCF in one of two ways, selected by its compiler attribute
:ref:`savecompiled<compiler-savecompiled>`:

- "true", the default: the values which the last compilation made of the model, in the canonical notation. For
  example a body which is :ref:`aligned<body-freejoint-align>` with its free joint is saved with its aligned pose, and
  a geom which was fitted to a mesh with the size of the fit. The spec must have been compiled.
- "false": the model as it is written in the spec. Poses are those of the elements in their :ref:`frames<frame>`,
  attributes which were written with their default values are kept, and every number is saved exactly. Orientations,
  angles and ``fromto`` are saved in the notation in which they were written, unless
  :ref:`savecanonical<compiler-savecanonical>` is "true". The spec need not have been compiled.

In both cases the saved file compiles to the same model, except where MJCF cannot say what the spec holds. For example,
the elements of an attached model are compiled with its own compiler settings, while the saved file has one
:ref:`compiler<compiler>` element. Saved as written, the angle unit and the Euler sequence of an attached model are
converted, and a difference in a setting which changes what compilation infers, such as
:ref:`inertiafromgeom<compiler-inertiafromgeom>`, is an error which says so; saved as compiled values, such a model may
compile differently. What the parser does not keep in a spec is not saved: comments other than the one at the top of
the file, the :ref:`include<include>` elements, and the elements which it expands into others, such as
:ref:`replicate<replicate>`, :ref:`flexcomp<body-flexcomp>`, :ref:`attach<body-attach>` and the actuator shortcuts.

A spec which is saved as written needs a compilation in two cases. The inertials which compilation calculates,
which :ref:`saveinertial<compiler-saveinertial>` saves, are those of the last compilation, which must follow any
structural edit (elements added, deleted or attached), as for compiled values. And :ref:`keyframes<keyframe>` which
await compilation, as they do once :ref:`mjs_attach` or :ref:`mjs_delete` changed the kinematic tree of a spec which
has keyframes, are saved once compilation has assembled their vectors.

Importantly, saved XML will take into account any defined defaults. This is useful when a model has many repeated
values, for example if loaded from URDF, which does not support defaults. In such a case one can add default classes,
set the class of the relevant elements, and save; the resulting XML will use the defaults and be more human-readable.

.. _MJZArchives:

MJZ Archives
~~~~~~~~~~~~

Complex MuJoCo models often consist of multiple files: a main MJCF XML file, included XML sub-trees, and external asset
files (meshes, textures, height fields). The **MJZ** format (extension ``.mjz`` or ``.zip``) provides a convenient way
to bundle an entire model and all of its referenced assets into a single **Zip archive**.

Root XML Discovery
^^^^^^^^^^^^^^^^^^

When decoding an ``.mjz`` archive, MuJoCo searches for the root model XML file in the following order:

1. ``model.xml`` at the root of the archive. This is the default format produced by the **MJZ**
   :ref:`encoder <exEncoder>` and is the most portable option.
2. ``<archive_stem>.xml`` at the root of the archive (e.g. ``my_model.xml`` inside ``my_model.mjz``).
3. ``<archive_stem>/<archive_stem>.xml`` inside a top-level directory matching the archive name (e.g.
   ``my_model/my_model.xml``).
4. ``<archive_stem>/model.xml`` inside a top-level directory matching the archive name (e.g. ``my_model/model.xml``).

Options 2-4 exist for backwards compatibility with legacy archives and common zip packaging layouts, but are
susceptible to failure if the archive file is renamed.

.. _meCustomFormats:

Custom formats
~~~~~~~~~~~~~~
Adding support for new file formats can be done with :ref:`mjp_registerDecoder` and :ref:`mjp_registerEncoder`.
When :ref:`mj_parse` and :ref:`mj_encode` are called for a non-native extension or content type, the appropriate plugins
are found via :ref:`mjp_findDecoder` and :ref:`mjp_findEncoder`. For further details on writing custom format plugins,
see :ref:`Decoders <exDecoder>` and :ref:`Encoders <exEncoder>`.


.. _meUsage:

Usage
~~~~~

Here we describe the C API for procedural model editing, but it is also exposed in the :ref:`Python
bindings<PyModelEdit>`. Advanced users can refer to `user_api_test.cc
<https://github.com/google-deepmind/mujoco/blob/main/test/user/user_api_test.cc>`__ and the MJCF parser in
`xml_native_reader.cc <https://github.com/google-deepmind/mujoco/blob/main/src/xml/xml_native_reader.cc>`__ for
more usage examples. After creating a new :ref:`mjSpec` or parsing an existing XML file to an :ref:`mjSpec`,
procedural editing corresponds to setting attributes. For example, in order to change the timestep, one can do:

.. code-block:: C

   mjSpec* spec = mj_makeSpec();
   spec->opt.timestep = 0.01;
   ...
   mjModel* model = mj_compile(spec, NULL);

Attributes which have variable length are C++ vectors and strings, :ref:`exposed to C as opaque types<ArrayHandles>`.
In C one uses the provided :ref:`getters<AttributeGetters>` and :ref:`setters<AttributeSetters>`:

.. code-block:: C

   mjs_setString(spec->modelname, "my_model");

In C++, one can use vectors and strings directly:

.. code-block:: C++

   std::string modelname = "my_model";
   *spec->modelname = modelname;

.. _meMjsElements:

Model elements
^^^^^^^^^^^^^^
Model elements corresponding to MJCF are exposed to the user as C structs with the ``mjs`` prefix. The definitions are
listed under the :ref:`Model Editing<tySpecStructure>` section of the struct reference. For example, an MJCF
:ref:`geom<body-geom>` corresponds to an :ref:`mjsGeom`.

Global defaults for all elements are set by :ref:`initializers<ElementInitialization>` like :ref:`mjs_defaultGeom`.
These functions are defined in `user_init.c
<https://github.com/google-deepmind/mujoco/blob/main/src/user/user_init.c>`__ and are the source of truth for all
default values.

Elements cannot be created directly; they are returned to the user by the corresponding constructor function, e.g.
:ref:`mjs_addGeom`. For example, to add a box geom to the world body, one would do

.. code-block:: C

   mjSpec* spec = mj_makeSpec();                                  // make an empty spec
   mjsBody* world = mjs_findBody(spec, "world");                  // find the world body
   mjsGeom* my_geom = mjs_addGeom(world, NULL);                   // add a geom to the world
   my_geom->type = mjGEOM_BOX;                                    // set geom type
   my_geom->size[0] = my_geom->size[1] = my_geom->size[2] = 0.5;  // set box size
   mjModel* model = mj_compile(spec, NULL);                       // compile to mjModel
   ...
   mj_deleteModel(model);                                         // free model
   mj_deleteSpec(spec);                                           // free spec

The ``NULL`` second argument to :ref:`mjs_addGeom` is the optional default class pointer. When using defaults
procedurally, default classes are passed in explicitly to element constructors. The global defaults of all elements
(used when no default class is passed in) can be inspected in
`user_init.c <https://github.com/google-deepmind/mujoco/blob/main/src/user/user_init.c>`__.

Elements are referenced by name, and names are set with :ref:`mjs_setName`. Assets are no exception: a mesh, texture or
height field added with :ref:`mjs_addMesh`, :ref:`mjs_addTexture` or :ref:`mjs_addHField` must be given a name, even
when it is loaded from a file. Naming an asset after its file when the name is omitted is a convenience of the XML
parser only.

.. _meMemory:

Memory management
^^^^^^^^^^^^^^^^^

As seen in the examples above, model elements are never allocated by the user directly, but rather returned by a
constructor. The library takes ownership of all elements and frees them when the parent :ref:`mjSpec` is deleted using
:ref:`mj_deleteSpec`. The user is only responsible for freeing :ref:`mjSpec` structs.

.. _meAttachment:

Attachment
^^^^^^^^^^

This framework introduces a powerful new feature: attaching and deleting model subtrees. This feature is already used to
power the :ref:`attach<body-attach>` and :ref:`replicate<replicate>` meta-elements in MJCF. Attachment allows the user to
move or copy a subtree from one model into another, while also copying or moving related referenced assets and
referencing elements from outside the kinematic tree (e.g., actuators and sensors). Similarly, deleting a subtree will
remove all associated elements from the model. The default behavior ("shallow copy") is to move the child into the
parent while attaching, so subsequent changes to the child will also change the parent. Alternatively, the user can
choose to make an entirely new copy during attach using :ref:`mjs_setDeepCopy`. This flag is temporarily set to true
while parsing XMLs. It is possible to :ref:`attach a body or an mjSpec to a frame<mjs_attach>`:

.. code-block:: C

   mjSpec* parent = mj_makeSpec();
   mjSpec* child = mj_makeSpec();
   parent->compiler.degree = 0;
   child->compiler.degree = 1;
   mjsElement* frame = mjs_addFrame(mjs_findBody(parent, "world"), NULL)->element;
   mjsElement* body = mjs_addBody(mjs_findBody(child, "world"), NULL)->element;
   mjsBody* attached_body_1 = mjs_asBody(mjs_attach(frame, body, "attached-", "-1"));

or :ref:`attach a body or an mjSpec to a site<mjs_attach>`:

.. code-block:: C

   mjSpec* parent = mj_makeSpec();
   mjSpec* child = mj_makeSpec();
   mjsElement* site = mjs_addSite(mjs_findBody(parent, "world"), NULL)->element;
   mjsElement* body = mjs_addBody(mjs_findBody(child, "world"), NULL)->element;
   mjsBody* attached_body_2 = mjs_asBody(mjs_attach(site, body, "attached-", "-2"));

or :ref:`attach a frame or an mjSpec to a body<mjs_attach>`:

.. code-block:: C

   mjSpec* parent = mj_makeSpec();
   mjSpec* child = mj_makeSpec();
   mjsElement* body = mjs_addBody(mjs_findBody(parent, "world"), NULL)->element;
   mjsElement* frame = mjs_addFrame(mjs_findBody(child, "world"), NULL)->element;
   mjsFrame* attached_frame = mjs_asFrame(mjs_attach(body, frame, "attached-", "-1"));

Note that in the above examples, the parent and child models have different values for ``compiler.degree``,
corresponding to the :ref:`compiler/angle<compiler-angle>` attribute, specifying the units in which angles are
interpreted. Compiler flags are carried over during attachment, so the child model will be compiled using the child
flags, while the parent will be compiled using the parent flags.

Note also that once a child is attached by reference to a parent, the child cannot be compiled on its own or attached
again as a whole. Its bodies and frames which are not attached yet can be attached, if they contain nothing which is
attached already. An element of the parent, including one attached to it by reference, can be attached to the parent
only as a copy, see :ref:`mjs_setDeepCopy`.

.. admonition:: Known issues
   :class: note

   The following known limitations exist:

   - All assets from the child model will be copied in, whether they are referenced or not, if the parent and the child
     are not the same mjSpec.
   - A child attached by reference more than once moves its assets to the parent with the first attachment. The elements
     of a later attachment find them only if it has the same prefix and suffix.
   - Circular references are not checked for and will lead to infinite loops.
   - When attaching a model with :ref:`keyframes<keyframe>`, its keyframes are added to the parent right away, but
     their vectors, and those of the keyframes of the parent, are assembled for the combined model only by the next
     compilation. Until then they are empty.

.. _meAttributeMerging:

Attribute Merging
^^^^^^^^^^^^^^^^^

When attaching a child spec (or an element from a child spec) to a parent spec using :ref:`mjs_attach`, global
attributes from the child may conflict with those in the parent. A conflict occurs when both the parent and child
specify authored values for the same field and those values differ. Note that for XML-based models, explicitly writing a
value (even if it matches the default value) counts as authoring and can trigger conflicts. The
:ref:`compiler/conflict<compiler-conflict>` attribute controls how such conflicts are resolved. Fields where only one
side specifies an authored value never conflict.

:at-val:`warning` (default)
   Parent values take precedence. Whenever a conflict is detected, a warning is emitted but the parent value is not
   modified. This preserves the pre-existing attachment behavior.

:at-val:`merge`
   Attribute values are merged using a per-field strategy as described in the table below. When only the child specifies
   an authored value, it is adopted by the parent.

:at-val:`error`
   Any conflict results in a compile error. No values are modified.

The table below describes the per-field merge strategy used in :at-val:`merge` mode.

.. list-table:: Attribute Merging Behavior (:at-val:`merge` mode)
   :widths: 15 60 25
   :header-rows: 1
   :name: meAttributeMergingTable

   * - Behavior
     - Fields
     - Justification
   * - **Minimum**
     - **option**: :ref:`timestep<option-timestep>`, :ref:`tolerance<option-tolerance>`, :ref:`ls_tolerance<option-ls_tolerance>`,
       :ref:`noslip_tolerance<option-noslip_tolerance>`, :ref:`ccd_tolerance<option-ccd_tolerance>`,
       :ref:`sleep_tolerance<option-sleep_tolerance>`,
       |br| **visual**: :ref:`znear<visual-map-znear>`, :ref:`realtime<visual-global-realtime>`
     - Preserves precision and stability.
   * - **Maximum**
     - **option**: :ref:`iterations<option-iterations>`, :ref:`ls_iterations<option-ls_iterations>`,
       :ref:`noslip_iterations<option-noslip_iterations>`, :ref:`ccd_iterations<option-ccd_iterations>`,
       :ref:`sdf_iterations<option-sdf_iterations>`, :ref:`sdf_initpoints<option-sdf_initpoints>`,
       |br| **size**: :ref:`memory<size-memory>`, :ref:`nkey<size-nkey>`, :ref:`nuserdata<size-nuserdata>`,
       :ref:`nuser_body<size-nuser_body>`, :ref:`nuser_jnt<size-nuser_jnt>`, :ref:`nuser_geom<size-nuser_geom>`, :ref:`nuser_site<size-nuser_site>`,
       :ref:`nuser_cam<size-nuser_cam>`, :ref:`nuser_tendon<size-nuser_tendon>`, :ref:`nuser_actuator<size-nuser_actuator>`, :ref:`nuser_sensor<size-nuser_sensor>`
       |br| **visual**: :ref:`zfar<visual-map-zfar>`
     - Ensures sufficient resources and limits.
   * - **OR (union)**
     - **option**: :ref:`disableflags<option-flag>`, :ref:`enableflags<option-flag>`,
       :ref:`disableactuator<option-actuatorgroupdisable>`
     - Flags from both models are combined.
   * - **Error**
     - **option**: :ref:`gravity<option-gravity>`, :ref:`wind<option-wind>`, :ref:`magnetic<option-magnetic>`,
       :ref:`density<option-density>`, :ref:`viscosity<option-viscosity>`, :ref:`integrator<option-integrator>`,
       :ref:`cone<option-cone>`, :ref:`jacobian<option-jacobian>`, :ref:`solver<option-solver>`,
       :ref:`impratio<option-impratio>`,
       :ref:`o_margin<option-o_margin>`, :ref:`o_solref<option-o_solref>`, :ref:`o_solimp<option-o_solimp>`,
       :ref:`o_friction<option-o_friction>`
     - Raised if non-default values conflict.

.. _meDefault:

Default classes
^^^^^^^^^^^^^^^
Default classes are fully supported in the new API, however using them requires an understanding of how defaults
are implemented. As explained in the :ref:`Default settings <CDefault>` section, default classes are first loaded as a
tree of dummy elements, which are then used to initialize elements which reference them. When editing models with
defaults, this initialization is explicit:

.. code-block:: C

   mjSpec* spec = mj_makeSpec();
   mjsDefault* main = mjs_getSpecDefault(spec);
   main->geom.type = mjGEOM_BOX;
   mjsGeom* geom = mjs_addGeom(mjs_findBody(spec, "world"), main);

Importantly, changing a default class after it has been used to initialize elements will not change the properties of
already initialized elements.

.. admonition:: Possible future change
   :class: note

   The behavior described above, where defaults are only applied at initialization, is a remnant of the old, XML-only
   loading pipeline. A future API change could allow defaults to be changed and applied after initialization. If you
   think this feature is important to you, please let us know on GitHub.


.. _meRecompilation:

In-place recompilation
^^^^^^^^^^^^^^^^^^^^^^

Compilation with :ref:`mj_compile` can be called at any point to obtain a new mjModel instance. In contrast,
:ref:`mj_recompile` updates an existing mjModel and mjData pair in-place, while preserving the simulation state. This
allows model editing to occur **during simulation**, for example adding or removing bodies.
