"""Abstract base classes for representing sequences in Graphtage's intermediate representation."""

from abc import ABC, abstractmethod
from typing import Any, Callable, cast, Dict, Generic, Iterable, Iterator, List, Optional, Sequence, Tuple, \
    Type, TypeVar

from .bounds import Range, repeat_until_tightened
from .edits import AbstractCompoundEdit, Insert, Match, Remove
from .printer import Printer
from .tree import ContainerNode, Edit, EditedTreeNode, GraphtageFormatter, TreeNode


class SequenceEdit(AbstractCompoundEdit, ABC):
    """An edit type for sequence nodes."""
    def __init__(
            self,
            from_node: 'SequenceNode',
            *args,
            **kwargs
    ):
        """Initializes a sequence edit.

        Args:
            from_node: The node being edited.
            *args: The remainder of the arguments to be passed to :meth:`AbstractCompoundEdit.__init__`.
            **kwargs: The remainder of the keyword arguments to be passed to :meth:`AbstractCompoundEdit.__init__`.

        Raises:
            ValueError: If :obj:`from_node` is not an instance of :class:`SequenceNode`.

        """
        if not isinstance(from_node, SequenceNode):
            raise ValueError(f"from_node must be a SequenceNode, but {from_node!r} is {type(from_node)}!")
        super().__init__(from_node=from_node, *args, **kwargs)

    @property
    def sequence(self) -> 'SequenceNode':
        """Returns the sequence being edited.

        This is a convenience function solely to aid in automated type checking. It is equivalent to::

            typing.cast(SequenceNode, self.from_node)

        """
        return cast(SequenceNode, self.from_node)

    def print(self, formatter: GraphtageFormatter, printer: Printer):
        """Prints this edit.

        This is equivalent to::

            formatter.get_formatter(self.sequence)(printer, self.sequence)

        """
        formatter.get_formatter(self.sequence)(printer, self.sequence)


class FixedLengthSequenceEdit(SequenceEdit):
    """An edit for sequences that are the same length.

    This does not consider insertion or removal from the list.

    """
    def __init__(
            self,
            from_node: 'SequenceNode',
            to_node: 'SequenceNode'
    ):
        if len(from_node) != len(to_node):
            raise ValueError("Both sequence nodes must have the same number of elements!")

        self._pair_iter: Optional[Iterator[Tuple[TreeNode, TreeNode]]] = zip(from_node, to_node)

        self._sub_edits: List[Edit] = []

        self._pair_sizes: List[int] = [f.total_size + t.total_size + 1 for f, t in zip(from_node, to_node)]

        super().__init__(from_node=from_node, to_node=to_node)

    def edits(self) -> Iterator[Edit]:
        yield from self._sub_edits
        while True:
            edit = self._expand_next_pair()
            if edit is None:
                break
            else:
                yield edit

    def _expand_next_pair(self) -> Optional[Edit]:
        if self._pair_iter is None:
            return None
        try:
            next_from, next_to = next(self._pair_iter)
            edit = next_from.edits(next_to)
            self._sub_edits.append(edit)
            return edit
        except StopIteration:
            self._pair_iter = None
            return None

    def is_complete(self) -> bool:
        return self._pair_iter is None and all(edit.is_complete() for edit in self._sub_edits)

    @repeat_until_tightened
    def tighten_bounds(self) -> bool:
        while self._pair_iter is not None:
            next_edit = self._expand_next_pair()
            if next_edit is not None and next_edit.tighten_bounds():
                return True
        for edit in self._sub_edits:
            if edit.tighten_bounds():
                return True
        return False

    def bounds(self) -> Range:
        if len(self._sub_edits) == len(self.sequence):
            lb = 0
            ub = 0
            for edit in self._sub_edits:
                b = edit.bounds()
                lb += b.lower_bound
                ub += b.upper_bound
            return Range(lb, ub)

        lb = 0
        ub = self.from_node.total_size + self.to_node.total_size + 1
        for original_ub, edit in zip(self._pair_sizes, self._sub_edits):
            b = edit.bounds()
            if b.upper_bound > original_ub:
                raise ValueError(f"Upper bound of edit {edit} is greater than the cost of a Replace")
            lb += b.lower_bound
            ub -= original_ub - b.upper_bound
        return Range(lb, ub)


T = TypeVar('T', bound=Sequence[TreeNode])


class SequenceNode(ContainerNode, Generic[T], ABC):
    """A node representing a sequence, like a list, set, or dict."""

    def __init__(self, children: T):
        """Initializes a sequence node.

        Args:
            children: A sequence of :class:`TreeNodes`.
                This is assigned to the protected member :attr:`SequenceNode._children`.

        """
        self._children: T = children

    def __len__(self) -> int:
        """The number of children of this sequence.

        This is equivalent to::

            return len(self._children)

        """
        return len(self._children)

    def __iter__(self) -> Iterator[TreeNode]:
        """Iterates over this sequence's child nodes.

        This is equivalent to::

            return iter(self._children)

        """
        return iter(self._children)

    def __repr__(self):
        return f"{self.__class__.__name__}({self._children!r})"

    def __str__(self):
        return str(self._children)

    def calculate_total_size(self):
        """Calculates the total size of this sequence.

        This is equivalent to::

            return sum(c.total_size for c in self)

        """
        return sum(c.total_size for c in self)

    def __eq__(self, other):
        return isinstance(other, SequenceNode) and self._children == other._children

    def __hash__(self):
        return hash(self._children)

    @property
    @abstractmethod
    def container_type(self) -> Type[T]:
        """Returns the container type used to store :attr:`SequenceNode._children`.

        This is used for performing a deep copy of this node in the :meth:`SequenceNode.editable_dict` function.

        """
        raise NotImplementedError()

    def editable_dict(self) -> Dict[str, Any]:
        """Copies :obj:`self.__dict__`, calling :meth:`TreeNode.editable_dict` on all children.

        This is equivalent to::

            ret = dict(self.__dict__)
            ret['_children'] = self.container_type(n.make_edited() for n in self)
            return ret

        This is used by :meth:`SequenceNode.make_edited`.

        """
        ret = dict(self.__dict__)
        ret['_children'] = self.container_type(n.make_edited() for n in self)
        return ret

    def print(self, printer: Printer):
        """Prints a sequence node.

        By default, sequence nodes are printed like lists::

            SequenceFormatter('[', ']', ',').print(printer, self)

        """
        SequenceFormatter('[', ']', ',').print(printer, self)


class SequenceFormatter(GraphtageFormatter):
    """A formatter for sequence nodes and edits.

    This class will typically be subclassed so that its methods can be overridden to match the style of its parent
    formatter. For an example, see the implementation of :class:`graphtage.json.JSONListFormatter` and
    :class:`graphtage.json.JSONDictFormatter`.

    """

    is_partial = True
    """This is a partial formatter; it will not be automatically used in the :ref:`Formatting Protocol`."""

    def __init__(
            self,
            start_symbol: str,
            end_symbol: str,
            delimiter: str,
            delimiter_callback: Optional[Callable[[Printer], Any]] = None
    ):
        """Initializes a sequence formatter.

        Args:
            start_symbol: The symbol to print at the start of the sequence.
            end_symbol: The symbol to print at the end of the sequence.
            delimiter: A delimiter to print between items.
            delimiter_callback: A callback for when a delimiter is to be printed. If omitted, this defaults to::

                lambda p: p.write(delimiter)

        """
        self.start_symbol = start_symbol
        self.end_symbol = end_symbol
        self.delimiter = delimiter
        if delimiter_callback is None:
            self.delimiter_callback: Callable[[Printer], Any] = lambda p: p.write(delimiter)
        else:
            self.delimiter_callback: Callable[[Printer], Any] = delimiter_callback

    def item_newline(self, printer: Printer, is_first: bool = False, is_last: bool = False):
        """Called before each node is printed.

        This is also called one extra time after the last node, if there is at least one node printed.

        The default implementation is simply::

            printer.newline()

        """
        printer.newline()

    def items_indent(self, printer: Printer) -> Printer:
        """Returns a Printer context with an indentation.

        This is called as::

            with self.items_indent(printer) as p:

        immediately after the :attr:`self.start_symbol` is printed, but before any of the items have been printed.

        This default implementation is equivalent to::

            return printer.indent()

        """
        return printer.indent()

    def edit_print(self, printer: Printer, edit: Edit):
        """Called when the edit for an item is to be printed.

        If the :class:`SequenceNode` being printed either is not edited or has no edits, then the edit passed to this
        function will be a :class:`Match(child, child, 0)<graphtage.Match>`.

        This implementation simply delegates the print to the :ref:`Formatting Protocol`::

            self.print(printer, edit)

        """
        self.print(printer, edit)

    def print_SequenceNode(self, printer: Printer, node: SequenceNode):
        """Formats a sequence node.

        The protocol for this function is as follows:

        * Print :attr:`self.start_symbol<SequenceFormatter.start_symbol>`
        * With the printer returned by :meth:`self.items_indent<SequenceFormatter.items_indent>`:
            * For each :obj:`edit` in the sequence (or just a sequence of :class:`graphtage.Match` for each child, if the node is not edited):
                * Call :meth:`self.item_newline(printer, is_first=index == 0)<SequenceFormatter.item_newline>`
                * Call :meth:`self.edit_print(printer, edit)<SequenceFormatter.edit_print>`
        * If at least one edit was printed, then call
          :meth:`self.item_newline(printer, is_last=True)<SequenceFormatter.item_newline>`
        * Print :attr:`self.start_symbol<SequenceFormatter.end_symbol>`

        """
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
