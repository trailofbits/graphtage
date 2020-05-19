"""A :class:`graphtage.Filetype` for parsing, diffing, and rendering `CSV files`_.

.. _CSV files:
    https://en.wikipedia.org/wiki/Comma-separated_values

"""

import csv
from io import StringIO
from typing import Optional

from . import graphtage, json
from .json import JSONFormatter
from .printer import Printer
from .sequences import SequenceFormatter
from .tree import GraphtageFormatter, TreeNode


class CSVRow(graphtage.ListNode[TreeNode]):
    """A node representing a row of a CSV file."""
    def __bool__(self):
        return bool(self._children)


class CSVNode(graphtage.ListNode[CSVRow]):
    """A node representing zero or more CSV rows."""
    def __bool__(self):
        return bool(self._children) and any(self._children)

    def __eq__(self, other: 'CSVNode'):
        return self._children == other._children or (not self and not other)


def build_tree(path: str, options: Optional[graphtage.BuildOptions] = None, *args, **kwargs) -> CSVNode:
    """Constructs a :class:`CSVNode` from a CSV file.

    The file is parsed using Python's :func:`csv.reader`. The elements in each row are constructed by delegating to
    :func:`graphtage.json.build_tree`::

        CSVRow([json.build_tree(i, options=options) for i in row])

    Args:
        path: The path to the file to be parsed.
        options: Optional build options to pass on to :meth:`graphtage.json.build_tree`.
        *args: Any extra positional arguments are passed on to :func:`csv.reader`.
        **kwargs: Any extra keyword arguments are passed on to :func:`csv.reader`.

    Returns:
        CSVNode: The resulting CSV node object.

    """
    csv_data = []
    with open(path) as f:
        for row in csv.reader(f, *args, **kwargs):
            rowdata = [json.build_tree(i, options=options) for i in row]
            for col in rowdata:
                if isinstance(col, graphtage.StringNode):
                    col.quoted = False
            csv_data.append(CSVRow(rowdata))
    return CSVNode(csv_data)


class CSVRowFormatter(SequenceFormatter):
    """A formatter for CSV rows."""
    is_partial = True

    def __init__(self):
        """Initializes the formatter.

        Equivalent to::

            super().__init__('', '', ',')

        """
        super().__init__('', '', ',')

    def print_CSVRow(self, *args, **kwargs):
        """Prints a CSV row.

        Equivalent to::

            super().print_SequenceNode(*args, **kwargs)

        """
        super().print_SequenceNode(*args, **kwargs)

    def item_newline(self, printer: Printer, is_first: bool = False, is_last: bool = False):
        """An empty implementation, since each row should be printed as a single line."""
        pass


class CSVRows(SequenceFormatter):
    """A sub formatter for printing the sequence of rows in a CSV file."""
    is_partial = True

    sub_format_types = [CSVRowFormatter]

    def __init__(self):
        """Initializes the formatter.

        Equivalent to::

            super().__init__('', '', '')

        """
        super().__init__('', '', '')

    def print_CSVNode(self, *args, **kwargs):
        """Prints a CSV node.

        Equivalent to::

            super().print_SequenceNode(*args, **kwargs)

        """
        super().print_SequenceNode(*args, **kwargs)

    def item_newline(self, printer: Printer, is_first: bool = False, is_last: bool = False):
        """Prints a newline on all but the first and last items."""
        if not is_first:
            printer.newline()

    def items_indent(self, printer: Printer):
        """Returns :obj:`printer` because CSV rows do not need to be indented."""
        return printer


class CSVFormatter(GraphtageFormatter):
    """Top-level formatter for CSV files."""
    sub_format_types = [CSVRows, JSONFormatter]

    def print_LeafNode(self, printer: Printer, node: graphtage.LeafNode):
        """Prints a leaf node, which should always be a column in a CSV row.

        The node is escaped by first writing it to :func:`csv.writer`::

            csv.writer(...).writerow([node.object])

        """
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
    """The CSV filetype."""
    def __init__(self):
        """Initializes the CSV filetype.

        CSV identifies itself with the MIME types `csv` and `text/csv`.

        """
        super().__init__(
            'csv',
            'text/csv'
        )

    def build_tree(self, path: str, options: Optional[graphtage.BuildOptions] = None) -> TreeNode:
        """Equivalent to :func:`build_tree`"""
        return build_tree(path, options=options)

    def build_tree_handling_errors(self, path: str, options: Optional[graphtage.BuildOptions] = None) -> TreeNode:
        return self.build_tree(path=path, options=options)

    def get_default_formatter(self) -> CSVFormatter:
        return CSVFormatter.DEFAULT_INSTANCE
