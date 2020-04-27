import csv
from typing import Any, Dict, Iterable, Union

from . import graphtage, json
from .json import JSONFormatter
from .printer import Printer
from .sequences import SequenceFormatter
from .tree import EditedTreeNode, TreeNode


class CSVRow(graphtage.ListNode):
    pass


class CSVNode(graphtage.ListNode):
    def __init__(self, rows: Iterable[Iterable[TreeNode]]):
        if isinstance(self, EditedTreeNode):
            super().__init__(
                [
                    CSVRow(list(row)).make_edited() for row in rows
                ]
            )
        else:
            super().__init__(
                [
                    CSVRow(list(row)) for row in rows
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
            rowdata = [json.build_tree(i, allow_key_edits=allow_key_edits) for i in row]
            for col in rowdata:
                if isinstance(col, graphtage.StringNode):
                    col.quoted = False
            csv_data.append(rowdata)
    return CSVNode(csv_data)


class CSVRowFormatter(SequenceFormatter):
    def __init__(self):
        super().__init__('', '', ',')

    def print_CSVRow(self, *args, **kwargs):
        super().print_SequenceNode(*args, **kwargs)

    def item_newline(self, printer: Printer, is_first: bool = False, is_last: bool = False):
        pass


class CSVRows(SequenceFormatter):
    sub_format_types = [CSVRowFormatter]

    def __init__(self):
        super().__init__('', '', '\n', lambda p: p.newline())

    def print_CSVNode(self, *args, **kwargs):
        super().print_SequenceNode(*args, **kwargs)

    def item_newline(self, printer: Printer, is_first: bool = False, is_last: bool = False):
        pass


class CSVFormatter(JSONFormatter):
    sub_format_types = [CSVRows]


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

    def get_default_formatter(self) -> CSVFormatter:
        return CSVFormatter.DEFAULT_INSTANCE
