from typing import Optional

import toml

from . import json
from .graphtage import BuildOptions, Filetype
from .tree import TreeNode


def build_tree(path: str, options: Optional[BuildOptions]) -> TreeNode:
    with open(path, 'r') as f:
        return json.build_tree(toml.load(f), options)


class TOML(Filetype):
    """The TOML filetype."""
    def __init__(self):
        """Initializes the TOML filetype.

        TOML identifies itself with the MIME types `application/toml` and `text/toml`.

        """
        super().__init__(
            'toml',
            'application/toml',
            'text/toml'
        )

    def build_tree(self, path: str, options: Optional[BuildOptions] = None) -> TreeNode:
        """Equivalent to :func:`build_tree`"""
        return build_tree(path, options=options)

    def build_tree_handling_errors(self, path: str, options: Optional[BuildOptions] = None) -> TreeNode:
        return self.build_tree(path=path, options=options)

    def get_default_formatter(self) -> json.JSONFormatter:
        return json.JSONFormatter.DEFAULT_INSTANCE
