.. _Filetypes:

Defining New Filetypes
======================

Implementing support for a new Graphtage filetype entails extending the :class:`graphtage.Filetype` class. Subclassing :class:`graphtage.Filetype` automatically registers it with Graphtage.

Filetype Matching
-----------------

Input files are matched to an associated :class:`graphtage.Filetype` using MIME types. Each :class:`graphtage.Filetype` registers one or more MIME types for which it will be responsible. Input file MIME types are classified using the :mod:`mimetypes` module. Sometimes a filetype does not have a standardized MIME type or is not properly classified by the :mod:`mimetypes` module. For example, Graphtage's :class:`graphtage.pickle.Pickle` filetype has neither. You can add support for such a filetype as follows:

.. code-block:: python

    import mimetypes

    if '.pkl' not in mimetypes.types_map and '.pickle' not in mimetypes.types_map:
        mimetypes.add_type('application/x-python-pickle', '.pkl')
        mimetypes.suffix_map['.pickle'] = '.pkl'

Implementing a New Filetype
---------------------------

With the MIME type registered, here is a sketch of how one might define the Pickle filetype:

.. code-block:: python

    from graphtage import BuildOptions, Filetype, Formatter, TreeNode

    class Pickle(Filetype):
        def __init__(self):
            super().__init__(
                "pickle",                      # a unique identifier
                "application/python-pickle",   # the primary MIME type
                "application/x-python-pickle"  # an optional secondary MIME type
            )

        def build_tree(self, path: str, options: Optional[BuildOptions] = None) -> TreeNode:
            # return the root node of the tree built from the given pickle file

        def build_tree_handling_errors(self, path: str, options: Optional[BuildOptions] = None) -> Union[str, TreeNode]:
            # the same as the build_tree() function,
            # but on error return a string containing the error message
            #
            # for example:
            try:
                return self.build_tree(path=path, options=options)
            except PickleDecodeError as e:
                return f"Error deserializing {os.path.basename(path)}: {e!s}"

        def get_default_formatter(self) -> GraphtageFormatter:
            # return the formatter associated with this file type
