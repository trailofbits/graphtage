import csv

from . import graphtage, json
from .tree import TreeNode


def build_tree(path: str, allow_key_edits=True, *args, **kwargs):
    csv_data = []
    with open(path) as f:
        for row in csv.reader(f, *args, **kwargs):
            csv_data.append(row)
    return json.build_tree(csv_data, allow_key_edits=allow_key_edits)


class CSV(graphtage.Filetype):
    def __init__(self):
        super().__init__(
            'csv',
            'text/csv'
        )

    def build_tree(self, path: str, allow_key_edits: bool = True) -> TreeNode:
        return build_tree(path, allow_key_edits=allow_key_edits)

    def build_tree_handling_errors(self, path: str, allow_key_edits: bool = True) -> TreeNode:
        return self.build_tree(path=path, allow_key_edits=allow_key_edits)
