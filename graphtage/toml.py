import itertools
import os
from collections.abc import Iterator, Sequence

import toml

from . import json
from .edits import Replace
from .graphtage import BuildOptions, Filetype, KeyValuePairNode, LeafNode, MappingNode, StringFormatter, StringNode
from .printer import Printer
from .sequences import SequenceFormatter
from .tree import Edit, EditedTreeNode, GraphtageFormatter, TreeNode


def build_tree(path: str, options: BuildOptions | None) -> TreeNode:
    with open(path) as f:
        return json.build_tree(toml.load(f), options)


class TOMLListFormatter(SequenceFormatter):
    """A sub-formatter for TOML lists."""
    is_partial = True

    def __init__(self):
        """Initializes the TOML list formatter.

        Equivalent to::

            super().__init__('[', ']', ',')

        """
        super().__init__('[', ']', ',')

    def item_newline(self, printer: Printer, is_first: bool = False, is_last: bool = False):
        pass

    def print_ListNode(self, *args, **kwargs):
        """Prints a :class:`graphtage.ListNode`.

        Equivalent to::

            super().print_SequenceNode(*args, **kwargs)

        """
        super().print_SequenceNode(*args, **kwargs)

    print_UnorderedListNode = print_ListNode

    def print_SequenceNode(self, *args, **kwargs):
        """Prints a non-List sequence.

        This delegates to the parent formatter's implementation::

            self.parent.print(*args, **kwargs)

        which should invoke :meth:`TOMLFormatter.print`, thereby delegating to the
        :class:`TOMLInlineTableFormatter` in instances where a list contains a dict.

        """
        self.parent.print(*args, **kwargs)


def toml_dumps(obj) -> str:
    s = toml.dumps({'result': obj})
    expected_prefix = 'result = '
    expected_suffix = '\n'
    assert s.startswith(expected_prefix)
    assert s.endswith(expected_suffix)
    return s[len(expected_prefix):-len(expected_suffix)]


class TOMLStringFormatter(StringFormatter):
    """A TOML formatter for strings."""
    is_partial = True

    def escape(self, c: str) -> str:
        s = toml_dumps(c)
        if s.startswith('"') and s.endswith('"'):
            return s[1:-1]
        else:
            return s


class TOMLInlineTableFormatter(SequenceFormatter):
    """A sub-formatter for mappings that have nowhere to write a ``[table]`` header.

    A mapping only gets a header when it is reached by walking down from the document root. One that appears in a
    value position has no such header available: the replaced side of an edit is printed inside the ``key = value``
    line it belongs to, and a mapping nested in a list is printed inside the list. Both are written as TOML inline
    tables, ``{key = value, ...}``, so that the value stays attached to its key.

    """
    is_partial = True

    def __init__(self):
        """Initializes the TOML inline table formatter.

        Equivalent to::

            super().__init__('{', '}', ',')

        """
        super().__init__('{', '}', ',')

    def item_newline(self, printer: Printer, is_first: bool = False, is_last: bool = False):
        """Separates two entries with a single space, since an inline table is written on one line."""
        if not is_first and not is_last:
            printer.write(' ')

    def items_indent(self, printer: Printer) -> Printer:
        """Returns :obj:`printer` itself, since an inline table writes no newline for an indent to apply to."""
        return printer

    def print_MappingNode(self, *args, **kwargs):
        """Prints a :class:`graphtage.MappingNode`.

        Equivalent to::

            super().print_SequenceNode(*args, **kwargs)

        """
        super().print_SequenceNode(*args, **kwargs)

    def print_KeyValuePairNode(self, printer: Printer, node: KeyValuePairNode):
        """Prints one entry of an inline table.

        This is :meth:`TOMLFormatter.write_key_value_pair` without the newline that ends a top-level
        ``key = value`` line, which would otherwise break the inline table across lines.

        """
        self.parent.write_key_value_pair(printer, node)

    def print_SequenceNode(self, *args, **kwargs):
        """Prints a non-mapping sequence.

        This delegates to the parent formatter's implementation::

            self.parent.print(*args, **kwargs)

        which should invoke :meth:`TOMLFormatter.print`, thereby delegating to the :class:`TOMLListFormatter` in
        instances where an inline table contains a list.

        """
        self.parent.print(*args, **kwargs)


def is_table(kvp: KeyValuePairNode) -> bool:
    """Returns whether a key/value pair is written as a ``[table]`` section rather than as ``key = value``.

    Only a pair whose value is a mapping can have a section of its own, and only if no edit replaces that mapping
    with a value of another type. A section body has nowhere to write such a replacement, so classifying the pair as
    a section would drop its edit from the output entirely.

    Args:
        kvp: The key/value pair to classify.

    Returns:
        bool: :const:`True` if :obj:`kvp` is written as its own ``[table]`` section, and :const:`False` if it is
        written inline as ``key = value``.

    """
    if not isinstance(kvp.value, MappingNode):
        return False
    return not (isinstance(kvp.value, EditedTreeNode) and isinstance(kvp.value.edit, Replace))


def key_value_pairs(mapping: MappingNode) -> Iterator[KeyValuePairNode]:
    """Iterates over a mapping's key/value pairs, including any that an edit inserts into it.

    Args:
        mapping: The mapping whose pairs to enumerate.

    Returns:
        Iterator[KeyValuePairNode]: Every pair the mapping renders. An inserted pair belongs to the other document,
        so it is not among the mapping's own children.

    """
    inserted: Sequence[TreeNode] = ()
    if isinstance(mapping, EditedTreeNode):
        inserted = mapping.inserted
    return itertools.chain(mapping, inserted)


def writes_own_section(mapping: MappingNode) -> bool:
    """Returns whether a mapping is written as a section of its own rather than only as a name prefix.

    A mapping whose every pair is itself a ``[table]`` writes nothing between its own header and the first header
    below it, so the header is omitted and the sub-tables carry the whole dotted name. An empty mapping still needs
    a header of its own, since it would otherwise leave no trace in the output.

    Args:
        mapping: The mapping to classify.

    Returns:
        bool: :const:`True` if :obj:`mapping` writes a header and a body of its own.

    """
    pairs = list(key_value_pairs(mapping))
    return not pairs or any(not is_table(kvp) for kvp in pairs)


class TOMLTableFormatter(SequenceFormatter):
    """A sub-formatter for a TOML document and for every ``[table]`` section within it.

    Every pair is written through :meth:`SequenceFormatter.print_SequenceNode`, which is where an inserted or removed
    pair is wrapped in its edit markup. A pair that becomes a section of its own is held back until the rest of its
    table has been written, because TOML reads every ``key = value`` line that follows a header as part of that
    header's table.

    """
    is_partial = True

    def __init__(self):
        """Initializes the TOML table formatter.

        Equivalent to::

            super().__init__('', '', '')

        """
        super().__init__('', '', '')
        self._name: list[TreeNode] = []
        self._sections: list[list[Edit]] = []

    def item_newline(self, printer: Printer, is_first: bool = False, is_last: bool = False):
        """Writes nothing, since every line of a table body already ends itself."""

    def items_indent(self, printer: Printer) -> Printer:
        """Returns :obj:`printer` itself, since TOML does not indent a table body under its header."""
        return printer

    def edit_print(self, printer: Printer, edit: Edit):
        """Writes one pair of a table, holding back any pair that becomes a ``[table]`` section of its own.

        Args:
            printer: The printer to which to write.
            edit: The edit for the pair, which is a :class:`graphtage.Match` when the table is not edited.

        """
        node = edit.from_node
        if isinstance(node, KeyValuePairNode) and is_table(node):
            self._sections[-1].append(edit)
        else:
            super().edit_print(printer, edit)

    def print_MappingNode(self, printer: Printer, node: MappingNode):
        """Writes a TOML document, or one ``[table]`` section and the sections nested within it.

        Args:
            printer: The printer to which to write.
            node: The document root, or the value of the pair that names this section.

        """
        writes_section = writes_own_section(node)
        if writes_section and self._name:
            self.write_header(printer)
        self._sections.append([])
        try:
            super().print_SequenceNode(printer, node)
        finally:
            sections = self._sections.pop()
        if writes_section:
            printer.newline()
        for section in sections:
            super().edit_print(printer, section)

    def print_KeyValuePairNode(self, printer: Printer, node: KeyValuePairNode):
        """Writes one entry of a table, either as a ``key = value`` line or as a ``[table]`` section of its own.

        Args:
            printer: The printer to which to write.
            node: The key/value pair to write.

        """
        if not is_table(node):
            self.parent.write_key_value_pair(printer, node)
            printer.newline()
            return
        self._name.append(node.key)
        try:
            self.print(printer, node.value)
        finally:
            self._name.pop()

    def print_SequenceNode(self, *args, **kwargs):
        """Prints a non-mapping sequence.

        This delegates to the parent formatter's implementation::

            self.parent.print(*args, **kwargs)

        which should invoke :meth:`TOMLFormatter.print`, thereby delegating to the :class:`TOMLListFormatter` in
        instances where a table contains a list.

        """
        self.parent.print(*args, **kwargs)

    def write_header(self, printer: Printer):
        """Writes the ``[dotted.name]`` header of the section that is currently being written.

        Args:
            printer: The printer to which to write.

        """
        printer.write('[')
        for i, segment in enumerate(self._name):
            if i > 0:
                printer.write('.')
            if isinstance(segment, StringNode):
                segment.quoted = False
            self.print(printer, segment)
        printer.write(']')
        printer.newline()


class TOMLFormatter(GraphtageFormatter):
    sub_format_types = (TOMLTableFormatter, TOMLListFormatter, TOMLStringFormatter, TOMLInlineTableFormatter)

    @property
    def inline_tables(self) -> TOMLInlineTableFormatter:
        """The sub-formatter for a mapping that has nowhere to write a ``[table]`` header.

        The :ref:`Formatting Protocol` resolves a formatter from a node's type alone, and a mapping written as a
        section and a mapping written inline are the same type, so this formatter selects between the two itself.

        """
        return next(f for f in self.sub_formatters if isinstance(f, TOMLInlineTableFormatter))

    @property
    def tables(self) -> TOMLTableFormatter:
        """The sub-formatter for a mapping that is written as a ``[table]`` section."""
        return next(f for f in self.sub_formatters if isinstance(f, TOMLTableFormatter))

    def print(self, printer: Printer, *args, **kwargs):
        # TOML has optional indentation; make it only two spaces, if we use it:
        printer.indent_str = '  '
        super().print(printer, *args, **kwargs)

    def print_LeafNode(self, printer: Printer, node: LeafNode):
        printer.write(toml_dumps(node.object))

    def write_key_value_pair(self, printer: Printer, node: KeyValuePairNode):
        """Writes ``key = value``, without the newline that ends a line of a table body.

        Args:
            printer: The printer to which to write.
            node: The key/value pair to write.

        """
        if isinstance(node.key, StringNode):
            node.key.quoted = False
        self.print(printer, node.key)
        printer.write(' = ')
        if isinstance(node.value, StringNode):
            node.value.quoted = True
        self.print(printer, node.value)

    def print_KeyValuePairNode(self, printer: Printer, node: KeyValuePairNode):
        self.write_key_value_pair(printer, node)
        printer.newline()

    def print_MappingNode(self, printer: Printer, node: MappingNode):
        if node.parent is not None:
            # This mapping is a value rather than the document, so there is no header to write it under:
            self.inline_tables.print_MappingNode(printer, node)
        else:
            self.tables.print_MappingNode(printer, node)


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

    def build_tree(self, path: str, options: BuildOptions | None = None) -> TreeNode:
        """Equivalent to :func:`build_tree`"""
        return build_tree(path, options=options)

    def build_tree_handling_errors(self, path: str, options: BuildOptions | None = None) -> str | TreeNode:
        try:
            return self.build_tree(path=path, options=options)
        except (IndexError, TypeError, ValueError) as e:
            return f'Error parsing {os.path.basename(path)}: {e})'

    def get_default_formatter(self) -> json.JSONFormatter:
        return TOMLFormatter.DEFAULT_INSTANCE
