import os
import sys

from yaml import load, YAMLError
try:
    from yaml import CLoader as Loader, CDumper as Dumper
except ImportError:
    from yaml import Loader, Dumper

from . import graphtage, json, TreeNode


def build_tree(path: str, allow_key_edits=True, *args, **kwargs) -> graphtage.TreeNode:
    with open(path, 'rb') as stream:
        data = load(stream, Loader=Loader)
        return json.build_tree(data, allow_key_edits=allow_key_edits, *args, **kwargs)


class YAML(graphtage.Filetype):
    def __init__(self):
        super().__init__(
            'yaml',
            'application/x-yaml',
            'application/yaml',
            'text/yaml',
            'text/x-yaml',
            'text/vnd.yaml'
        )

    def build_tree(self, path: str, allow_key_edits: bool = True) -> TreeNode:
        return build_tree(path=path, allow_key_edits=allow_key_edits)

    def build_tree_handling_errors(self, path: str, allow_key_edits: bool = True) -> TreeNode:
        try:
            return self.build_tree(path=path, allow_key_edits=allow_key_edits)
        except YAMLError as ye:
            sys.stderr.write(f'Error parsing {os.path.basename(path)}: {ye})\n\n')
            sys.exit(1)
