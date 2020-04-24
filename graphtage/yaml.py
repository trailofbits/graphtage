from yaml import load
try:
    from yaml import CLoader as Loader, CDumper as Dumper
except ImportError:
    from yaml import Loader, Dumper

from . import graphtage


def build_tree(path: str, allow_key_edits=True, *args, **kwargs) -> graphtage.TreeNode:
    with open(path, 'rb') as stream:
        data = load(stream, Loader=Loader)
        return graphtage.build_tree(data, allow_key_edits=allow_key_edits, *args, **kwargs)
