import itertools
import os
from typing import Iterator, Optional, Tuple, Union

import toml

from . import json
from .graphtage import BuildOptions, Filetype, KeyValuePairNode, LeafNode, MappingNode, StringFormatter, StringNode
from .printer import Printer
from .sequences import SequenceFormatter
from .tree import GraphtageFormatter, TreeNode


def build_tree(path: str, options: Optional[BuildOptions]) -> TreeNode:
    with open(path, 'r') as f:
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

    def print_SequenceNode(self, *args, **kwargs):
        """Prints a non-List sequence.

        This delegates to the parent formatter's implementation::

            self.parent.print(*args, **kwargs)

        which should invoke :meth:`TOMLFormatter.print`, thereby delegating to the :class:`TOMLDictFormatter` in
        instances where a list contains a dict (the TOML format doesn't allow this, but it might be necessary if
        formatting from another format into TOML)

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


class TOMLMapping:
    def __init__(
            self,
            mapping: MappingNode,
            parent: Optional['TOMLMapping'] = None,
            parent_name: Optional[TreeNode] = None
    ):
        self.mapping: MappingNode = mapping
        self.parent: Optional[TOMLMapping] = parent
        self.parent_name: Optional[TreeNode] = parent_name

    @property
    def name_segments(self) -> Tuple[TreeNode, ...]:
        if self.parent is None:
            return ()
        else:
            return self.parent.name_segments + (self.parent_name,)

    def items(self) -> Iterator[KeyValuePairNode]:
        inserted = ()
        if self.mapping.edited and self.mapping.inserted:
            inserted = self.mapping.inserted
        for kvp in itertools.chain(self.mapping, inserted):
            if not isinstance(kvp.value, MappingNode):
                yield kvp

    def __bool__(self):
        try:
            next(self.items())
            return True
        except StopIteration:
            try:
                next(self.children())
                return False
            except StopIteration:
                return True

    def children(self) -> Iterator['TOMLMapping']:
        inserted = ()
        if self.mapping.edited and self.mapping.inserted:
            inserted = self.mapping.inserted
        for kvp in itertools.chain(self.mapping, inserted):
            if isinstance(kvp.value, MappingNode):
                yield TOMLMapping(mapping=kvp.value, parent=self, parent_name=kvp.key)


class TOMLFormatter(GraphtageFormatter):
    sub_format_types = [TOMLListFormatter, TOMLStringFormatter]

    def print(self, printer: Printer, *args, **kwargs):
        # TOML has optional indentation; make it only two spaces, if we use it:
        printer.indent_str = '  '
        super().print(printer, *args, **kwargs)

    def print_LeafNode(self, printer: Printer, node: LeafNode):
        printer.write(toml_dumps(node.object))

    def print_KeyValuePairNode(self, printer: Printer, node: KeyValuePairNode):
        if isinstance(node.key, StringNode):
            node.key.quoted = False
        self.print(printer, node.key)
        printer.write(' = ')
        if isinstance(node.value, StringNode):
            node.value.quoted = True
        self.print(printer, node.value)
        printer.newline()

    def print_MappingNode(self, printer: Printer, node: MappingNode):
        mappings = [TOMLMapping(node)]
        while mappings:
            m: TOMLMapping = mappings.pop()
            if m:
                name = m.name_segments
                if name:
                    printer.write('[')
                    first = True
                    for s in name:
                        if first:
                            first = False
                        else:
                            printer.write('.')
                        if isinstance(s, StringNode):
                            s.quoted = False
                        self.print(printer, s)
                    printer.write(']')
                    printer.newline()
                for kvp in m.items():
                    self.print(printer, kvp)
                printer.newline()
            mappings.extend(m.children())


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

    def build_tree(self, path: str, options: Optional[BuildOptions] = None) -> TreeNode:
        """Equivalent to :func:`build_tree`"""
        return build_tree(path, options=options)

    def build_tree_handling_errors(self, path: str, options: Optional[BuildOptions] = None) -> Union[str, TreeNode]:
        try:
            return self.build_tree(path=path, options=options)
        except (IndexError, TypeError, ValueError) as e:
            return f'Error parsing {os.path.basename(path)}: {e})'

    def get_default_formatter(self) -> json.JSONFormatter:
        return TOMLFormatter.DEFAULT_INSTANCE
