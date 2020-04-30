import csv
from typing import Any, Dict, Iterable, Union

from . import graphtage, json
from .json import JSONFormatter
from .printer import Printer
from .sequences import SequenceFormatter
from .tree import TreeNode


class CSVRow(graphtage.ListNode[TreeNode]):
    pass


class CSVNode(graphtage.ListNode[CSVRow]):
    pass


def build_tree(path: str, allow_key_edits=True, *args, **kwargs) -> TreeNode:
    csv_data = []
    with open(path) as f:
        for row in csv.reader(f, *args, **kwargs):
            rowdata = [json.build_tree(i, allow_key_edits=allow_key_edits) for i in row]
            for col in rowdata:
                if isinstance(col, graphtage.StringNode):
                    col.quoted = False
            csv_data.append(CSVRow(rowdata))
    return CSVNode(csv_data)


class CSVRowFormatter(SequenceFormatter):
    is_partial = True

    def __init__(self):
        super().__init__('', '', ',')

    def print_CSVRow(self, *args, **kwargs):
        super().print_SequenceNode(*args, **kwargs)

    def item_newline(self, printer: Printer, is_first: bool = False, is_last: bool = False):
        pass


class CSVRows(SequenceFormatter):
    is_partial = True

    sub_format_types = [CSVRowFormatter]

    def __init__(self):
        super().__init__('', '', '')

    def print_CSVNode(self, *args, **kwargs):
        super().print_SequenceNode(*args, **kwargs)

    def item_newline(self, printer: Printer, is_first: bool = False, is_last: bool = False):
        if not is_first and not is_last:
            printer.newline()

    def items_indent(self, printer: Printer):
        return printer


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
