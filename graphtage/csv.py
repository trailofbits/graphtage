import csv
from typing import Any, Dict, Iterable, Union

from . import graphtage, json
from .printer import Printer
from .tree import EditedTreeNode, TreeNode


class CSVNode(graphtage.ListNode):
    def __init__(self, rows: Iterable[Iterable[TreeNode]]):
        if isinstance(self, EditedTreeNode):
            super().__init__(
                [
                    graphtage.ListNode(list(row)).make_edited() for row in rows
                ]
            )
        else:
            super().__init__(
                [
                    graphtage.ListNode(list(row)) for row in rows
                ]
            )
        self.start_symbol = ''
        self.end_symbol = ''
        self.delimiter_callback = lambda p: p.newline()

    def print_item_newline(self, printer: Printer, is_first: bool = False, is_last: bool = False):
        printer.newline()

    def init_args(self) -> Dict[str, Any]:
        return {
            'rows': [
                row.children for row in self.children
            ]
        }

    def make_edited(self) -> Union[EditedTreeNode, 'CSVNode']:
        return self.copy(new_class=self.edited_type())


def build_tree(path: str, allow_key_edits=True, *args, **kwargs) -> TreeNode:
    csv_data = []
    with open(path) as f:
        for row in csv.reader(f, *args, **kwargs):
            csv_data.append([json.build_tree(i, allow_key_edits=allow_key_edits) for i in row])
    return CSVNode(csv_data)


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
