import os
from typing import Optional, Union

from fickling.fickle import Interpreter, Pickled, PickleDecodeError

from .graphtage import BuildOptions, Filetype, TreeNode
from .pydiff import ast_to_tree, PyDiffFormatter


class Pickle(Filetype):
    """The Python Pickle file type."""
    def __init__(self):
        """Initializes the Pickle file type.

        By default, Pickle associates itself with the "pickle", "application/python-pickle",
        and "application/x-python-pickle" MIME types.

        """
        super().__init__(
            'pickle',
            'application/python-pickle',
            'application/x-python-pickle'
        )

    def build_tree(self, path: str, options: Optional[BuildOptions] = None) -> TreeNode:
        with open(path, "rb") as f:
            pickle = Pickled.load(f)
            interpreter = Interpreter(pickle)
            ast = interpreter.to_ast()
            return ast_to_tree(ast, options)

    def build_tree_handling_errors(self, path: str, options: Optional[BuildOptions] = None) -> Union[str, TreeNode]:
        try:
            return self.build_tree(path=path, options=options)
        except PickleDecodeError as e:
            return f'Error deserializing {os.path.basename(path)}: {e!s}'

    def get_default_formatter(self) -> PyDiffFormatter:
        return PyDiffFormatter.DEFAULT_INSTANCE
