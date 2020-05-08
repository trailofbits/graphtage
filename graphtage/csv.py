import csv
from io import StringIO

from . import graphtage, json
from .json import JSONFormatter
from .printer import Printer
from .sequences import SequenceFormatter
from .tree import GraphtageFormatter, TreeNode


class CSVRow(graphtage.ListNode[TreeNode]):
    def __bool__(self):
        return bool(self._children)


class CSVNode(graphtage.ListNode[CSVRow]):
    def __bool__(self):
        return bool(self._children) and any(self._children)

    def __eq__(self, other: 'CSVNode'):
        return self._children == other._children or (not self and not other)


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


class CSVFormatter(GraphtageFormatter):
    sub_format_types = [CSVRows, JSONFormatter]

    def print_LeafNode(self, printer: Printer, node: graphtage.LeafNode):
        if node.edited and node.edit is not None:
            self.sub_formatters[1].print(printer, node.edit)
            return
        s = StringIO()
        writer = csv.writer(s)
        writer.writerow([node.object])
        r = s.getvalue()
        if r.endswith('\r\n'):
            r = r[:-2]
        elif r.endswith('\n') or r.endswith('\r'):
            r = r[:-1]
        printer.write(r)
        s.close()


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
