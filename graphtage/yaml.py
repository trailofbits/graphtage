from yaml import load
try:
    from yaml import CLoader as Loader, CDumper as Dumper
except ImportError:
    from yaml import Loader, Dumper

from . import graphtage


def build_tree(path: str) -> graphtage.TreeNode:
    with open(path, 'rb') as stream:
        data = load(stream, Loader=Loader)
        return graphtage.build_tree(data)
