.. _Filetypes:

Defining New Filetypes
======================

Implementing support for a new Graphtage filetype entails extending the :class:`graphtage.Filetype` class. Subclassing :class:`graphtage.Filetype` automatically registers it with Graphtage.

Subclassing is only the first of several steps, though. A filetype that Graphtage can name but cannot recognize on disk, or one whose formatter cannot resolve every node type it emits, fails at runtime rather than at import. The sections below cover each step in turn.

Filetype Matching
-----------------

Input files are matched to an associated :class:`graphtage.Filetype` using MIME types. Each :class:`graphtage.Filetype` registers one or more MIME types for which it will be responsible. Input file MIME types are classified using the :mod:`mimetypes` module: :func:`graphtage.get_filetype` calls :func:`mimetypes.guess_type` on the path and looks the result up in :data:`graphtage.FILETYPES_BY_MIME`.

Sometimes a filetype does not have a standardized MIME type or is not properly classified by the :mod:`mimetypes` module. For example, Graphtage's :class:`graphtage.pickle.Pickle` filetype has neither. Graphtage registers the missing types in :func:`graphtage.__main__.register_mimetypes`, which the command line entry point calls once, after it has parsed its arguments and therefore after every filetype module has been imported:

.. code-block:: python

    def register_mimetypes():
        mimetypes.init()
        ...
        if '.pkl' not in mimetypes.types_map and '.pickle' not in mimetypes.types_map:
            mimetypes.add_type('application/x-python-pickle', '.pkl')
            mimetypes.suffix_map['.pickle'] = '.pkl'

The first line is the reason a filetype cannot register its own extension at import time. :func:`mimetypes.init` rebuilds the module's global ``types_map`` from scratch, discarding anything an earlier :func:`mimetypes.add_type` call added. A module that calls :func:`mimetypes.add_type` when it is imported therefore loses its registration as soon as :func:`register_mimetypes` runs, and every diff of that filetype fails with ``Could not determine the filetype``.

There are two places to register an extension, depending on where your filetype lives:

* A filetype inside Graphtage adds its extension to :func:`graphtage.__main__.register_mimetypes`, alongside the ones already there.
* A filetype outside Graphtage calls :func:`mimetypes.add_type` after :func:`graphtage.__main__.register_mimetypes` has run. Calling it while your module is imported is not enough. From your own entry point, call :func:`graphtage.__main__.register_mimetypes` first and add your type afterward:

  .. code-block:: python

      from graphtage.__main__ import register_mimetypes

      register_mimetypes()
      mimetypes.add_type("application/x-widget", ".widget")

  Under the ``graphtage`` command, where you have no chance to run code between the two calls, name the type explicitly with the ``--from-mime`` and ``--to-mime`` options, which skip the guess entirely.

Implementing a New Filetype
---------------------------

With the MIME type registered, here is how the Pickle filetype is defined. This is :mod:`graphtage.pickle`, with its docstrings removed and its imports written as an out-of-tree filetype would write them:

.. code-block:: python

    import os

    from fickling.fickle import Interpreter, Pickled, PickleDecodeError

    from graphtage import BuildOptions, Filetype, TreeNode
    from graphtage.pydiff import PyDiffFormatter, ast_to_tree


    class Pickle(Filetype):
        def __init__(self):
            super().__init__(
                "pickle",                       # a unique identifier
                "application/python-pickle",    # the primary MIME type
                "application/x-python-pickle",  # an optional secondary MIME type
            )

        def build_tree(self, path: str, options: BuildOptions | None = None) -> TreeNode:
            with open(path, "rb") as f:
                pickled = Pickled.load(f)
                ast = Interpreter(pickled).to_ast()
                return ast_to_tree(ast, options)

        def build_tree_handling_errors(self, path: str, options: BuildOptions | None = None) -> str | TreeNode:
            # the same as build_tree(), but on error return a string containing the error message
            try:
                return self.build_tree(path=path, options=options)
            except PickleDecodeError as e:
                return f"Error deserializing {os.path.basename(path)}: {e!s}"

        def get_default_formatter(self) -> PyDiffFormatter:
            return PyDiffFormatter.DEFAULT_INSTANCE

:class:`graphtage.FiletypeWatcher` instantiates the subclass at class definition time, so :meth:`__init__<graphtage.Filetype.__init__>` must take no arguments beyond ``self``, and all three methods must be implemented. Registration into :data:`graphtage.FILETYPES_BY_TYPENAME` and :data:`graphtage.FILETYPES_BY_MIME` happens automatically, and that registration is what generates the ``--from-pickle``, ``--to-pickle``, and ``--format pickle`` command line options.

Because the class statement itself creates and registers the instance, retrieve your filetype from the registry rather than constructing it:

.. code-block:: python

    filetype = graphtage.FILETYPES_BY_TYPENAME["pickle"]
    tree = filetype.build_tree("example.pkl")

Calling ``Pickle()`` a second time raises ``ValueError: MIME type application/python-pickle is already assigned``. Running the snippet above verbatim raises the same error, since Graphtage already ships :mod:`graphtage.pickle` under those three identifiers; change the type name and MIME types to try it out.

If your parser already produces plain Python objects, :func:`graphtage.json.build_tree` converts one into a tree that satisfies the whole :class:`graphtage.TreeNode` contract, so :meth:`build_tree<graphtage.Filetype.build_tree>` can be a single call.

Registering the Module
----------------------

Defining the class is not enough on its own: nothing imports it. A filetype inside Graphtage is added to the ``from . import ...`` statement in ``graphtage/__init__.py``, which is also where ``docs/build_api.py`` discovers the modules it generates API pages for. A filetype outside Graphtage must be imported by whatever code runs the diff, before :func:`graphtage.get_filetype` is called.

Writing the Formatter
---------------------

:meth:`Filetype.get_default_formatter<graphtage.Filetype.get_default_formatter>` returns the :class:`graphtage.GraphtageFormatter` that renders your nodes. Four rules govern how formatters compose; see :mod:`graphtage.json` and :mod:`graphtage.ini` for worked examples.

**Route sequence printing through** :meth:`SequenceFormatter.print_SequenceNode<graphtage.sequences.SequenceFormatter.print_SequenceNode>`. That method is where insertion and removal edits are turned into output. A formatter that iterates a node's children directly renders the unedited tree correctly and silently drops every insertion and removal, so define ``print_ListNode`` and its siblings as thin wrappers that call ``super().print_SequenceNode(*args, **kwargs)``. A sub-formatter overrides ``print_SequenceNode`` itself to hand any other sequence back to the parent formatter with ``self.parent.print(*args, **kwargs)``, which is how a list containing a dictionary reaches the dictionary sub-formatter.

**Alias** ``print_UnorderedListNode`` **to** ``print_ListNode``. A :class:`graphtage.UnorderedListNode` reaches your formatter whenever the user passes ``--ignore-list-order``, and it does not inherit from :class:`graphtage.ListNode`, so ``print_ListNode`` alone does not match it. Without the alias, the lookup walks the node's method resolution order down to :class:`graphtage.sequences.SequenceNode` and settles on ``print_SequenceNode``, which delegates back to the parent formatter, which resolves the node the same way again. The print recurses until the interpreter runs out of stack:

.. code-block:: python

    def print_ListNode(self, *args, **kwargs):
        super().print_SequenceNode(*args, **kwargs)

    print_UnorderedListNode = print_ListNode

**Mark helper formatters** ``is_partial = True``. :class:`graphtage.formatter.FormatterChecker` appends every non-partial formatter to the global :data:`graphtage.formatter.FORMATTERS` list, which :func:`graphtage.formatter.get_formatter` searches when no closer match is found. A helper that is registered globally can therefore be chosen to print a node in an unrelated format. Only the one formatter that :meth:`get_default_formatter<graphtage.Filetype.get_default_formatter>` returns should be non-partial.

**Define each** ``print_<NodeType>`` **method once per formatter tree.** :func:`graphtage.formatter.get_formatter` returns the first match it finds while walking a formatter, its sub-formatters, and then its parents, so a second definition of the same method elsewhere in the tree is unreachable. When two nesting levels need different output for the same node type, branch inside a single key/value formatter rather than splitting the work across two formatters by node type.

Testing a New Filetype
----------------------

Graphtage's own test suite requires a ``test_<typename>_formatting`` method in ``test/test_formatting.py`` for every registered filetype; ``test_formatter_coverage`` fails without one. The ``@filetype_test`` decorator only round-trips an *unedited* tree, so it cannot catch a formatter that mishandles edits. Add a separate test that diffs two documents and checks that insertions and removals appear in the output.
