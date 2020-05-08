from abc import ABC, abstractmethod
from typing import Any, Callable, cast, Dict, Generic, Iterable, Iterator, Optional, Sequence, Type, TypeVar

from .edits import AbstractCompoundEdit, Insert, Match, Remove
from .printer import Printer
from .tree import ContainerNode, Edit, EditedTreeNode, GraphtageFormatter, TreeNode


class SequenceEdit(AbstractCompoundEdit, ABC):
    def __init__(
            self,
            from_node,
            *args,
            **kwargs
    ):
        if not isinstance(from_node, SequenceNode):
            raise ValueError(f"from_node must be a SequenceNode, but {from_node!r} is {type(from_node)}!")
        super().__init__(from_node=from_node, *args, **kwargs)

    @property
    def sequence(self) -> 'SequenceNode':
        return cast(SequenceNode, self.from_node)

    def print(self, formatter: GraphtageFormatter, printer: Printer):
        formatter.get_formatter(self.sequence)(printer, self.sequence)


T = TypeVar('T', bound=Sequence[TreeNode])


class SequenceNode(ContainerNode, Generic[T], ABC):
    def __init__(self, children: T):
        self._children: T = children

    def __len__(self) -> int:
        return len(self._children)

    def __iter__(self) -> Iterator[TreeNode]:
        return iter(self._children)

    def __repr__(self):
        return f"{self.__class__.__name__}({self._children!r})"

    def __str__(self):
        return str(self._children)

    def calculate_total_size(self):
        return sum(c.total_size for c in self)

    def __eq__(self, other):
        return isinstance(other, SequenceNode) and self._children == other._children

    def __hash__(self):
        return hash(self._children)

    @property
    @abstractmethod
    def container_type(self) -> Type[T]:
        raise NotImplementedError()

    def editable_dict(self) -> Dict[str, Any]:
        ret = dict(self.__dict__)
        ret['_children'] = self.container_type(n.make_edited() for n in self)
        return ret

    def print(self, printer: Printer):
        SequenceFormatter('[', ']', ',').print(printer, self)


class SequenceFormatter(GraphtageFormatter):
    is_partial = True

    def __init__(
            self,
            start_symbol: str,
            end_symbol: str,
            delimiter: str,
            delimiter_callback: Optional[Callable[[Printer], Any]] = None
    ):
        self.start_symbol = start_symbol
        self.end_symbol = end_symbol
        self.delimiter = delimiter
        if delimiter_callback is None:
            self.delimiter_callback: Callable[[Printer], Any] = lambda p: p.write(delimiter)
        else:
            self.delimiter_callback: Callable[[Printer], Any] = delimiter_callback

    def item_newline(self, printer: Printer, is_first: bool = False, is_last: bool = False):
        printer.newline()

    def items_indent(self, printer: Printer):
        return printer.indent()

    def edit_print(self, printer: Printer, edit: Edit):
        self.print(printer, edit)

    def print_SequenceNode(self, printer: Printer, node: SequenceNode):
        with printer.bright():
            printer.write(self.start_symbol)
        with self.items_indent(printer) as p:
            to_remove: int = 0
            to_insert: int = 0
            if isinstance(node, EditedTreeNode) and isinstance(node.edit, SequenceEdit):
                edits: Iterable[Edit] = node.edit.edits()
            else:
                edits: Iterable[Edit] = [Match(child, child, 0) for child in node]
            for i, edit in enumerate(edits):
                if isinstance(edit, Remove):
                    to_remove += 1
                elif isinstance(edit, Insert):
                    to_insert += 1
                while to_remove > 0 and to_insert > 0:
                    to_remove -= 1
                    to_insert -= 1
                if i > 0:
                    with printer.bright():
                        if to_remove:
                            to_remove -= 1
                            with printer.strike():
                                self.delimiter_callback(printer)
                        elif to_insert:
                            to_insert -= 1
                            with printer.under_plus():
                                self.delimiter_callback(printer)
                        else:
                            self.delimiter_callback(p)
                self.item_newline(printer, is_first=i == 0)
                self.edit_print(printer, edit)
        if len(node) > 0:
            self.item_newline(printer, is_last=True)
        with printer.bright():
            printer.write(self.end_symbol)
