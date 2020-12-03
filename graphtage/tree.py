import itertools
import logging
import sys
from abc import abstractmethod, ABC, ABCMeta
from collections.abc import Iterable
from typing import Any, Callable, Collection, Dict, Iterator, List, Optional, Sized, Type, TypeVar, Union
from typing_extensions import Protocol, runtime_checkable

from .bounds import Bounded, Range
from .formatter import Formatter, FORMATTERS
from .printer import DEFAULT_PRINTER, Printer

log = logging.getLogger(__name__)


if sys.version_info.major == 3 and sys.version_info.minor < 7:
    # For some reason, the type checker breaks on the generic argument in Py3.6 and earlier
    FormatterType = Formatter
else:
    FormatterType = Formatter[Union['TreeNode', 'Edit']]


class GraphtageFormatter(FormatterType):
    """A base class for defining formatters that operate on :class:`TreeNode` and :class:`Edit`."""

    def print(self, printer: Printer, node_or_edit: Union['TreeNode', 'Edit'], with_edits: bool = True):
        """Prints the given node or edit.

        Args:
            printer: The printer to which to write.
            node_or_edit: The node or edit to print.
            with_edits: If :keyword:True, print any edits associated with the node.

        Note:
            The protocol for determining how a node or edit should be printed is very complex due to its extensibility.
            See the :ref:`Printing Protocol` for a detailed description.

        """
        if isinstance(node_or_edit, Edit):
            if with_edits:
                edit: Optional[Edit] = node_or_edit
            else:
                edit: Optional[Edit] = None
            node: TreeNode = node_or_edit.from_node
        elif with_edits:
            if isinstance(node_or_edit, EditedTreeNode) and \
                    node_or_edit.edit is not None and node_or_edit.edit.has_non_zero_cost():
                edit: Optional[Edit] = node_or_edit.edit
                node: TreeNode = node_or_edit
            else:
                edit: Optional[Edit] = None
                node: TreeNode = node_or_edit
        else:
            edit: Optional[Edit] = None
            node: TreeNode = node_or_edit
        if edit is not None:
            # First, see if we have a specialized formatter for this edit:
            edit_formatter = self.get_formatter(edit)
            if edit_formatter is not None:
                edit_formatter(printer, edit)
                return
            try:
                edit.print(self, printer)
                return
            except NotImplementedError:
                pass
        formatter = self.get_formatter(node)
        if formatter is not None:
            formatter(printer, node)
        else:
            log.debug(f"""There is no formatter that can handle nodes of type {node.__class__.__name__}
    Falling back to the node's internal printer
    Registered formatters: {''.join([f.__class__.__name__ for f in FORMATTERS])}""")
            node.print(printer)


@runtime_checkable
class Edit(Bounded, Protocol):
    """A protocol for defining an edit.

    Attributes:
        initial_bounds (Range): The initial bounds of this edit. Classes implementing this protocol can, for example,
            set this by calling :meth:`self.bounds()<Edit.bounds>` during initialization.

        from_node (TreeNode): The node that this edit transforms.

    """
    initial_bounds: Range
    """The initial bounds of this edit.

    Classes implementing this protocol can, for example, set this by calling
    :meth:`self.bounds()<Edit.bounds>` during initialization.

    """
    from_node: 'TreeNode'
    """The node that this edit transforms."""

    def has_non_zero_cost(self) -> bool:
        """Returns whether this edit has a non-zero cost.

        This will tighten the edit's bounds until either its lower bound is greater than zero or its bounds are
        definitive.

        """
        while not self.bounds().definitive() and self.bounds().lower_bound <= 0 and self.tighten_bounds():
            pass
        return self.bounds().lower_bound > 0

    @abstractmethod
    def bounds(self) -> Range:
        """The bounds on the cost of this edit.

        The lower bound must always be finite and non-negative.

        Returns:
            Range: The bounds on the cost of this edit.
        """
        raise NotImplementedError()

    @abstractmethod
    def tighten_bounds(self) -> bool:
        """Tightens the :meth:`Edit.bounds` on the cost of this edit, if possible.

        Returns:
            bool: :const:`True` if the bounds have been tightened.

        Note:
            Implementations of this function should return :const:`False` if and only if
            :meth:`self.bounds().definitive() <graphtage.bounds.Range.definitive>`.

        """
        raise NotImplementedError()

    @abstractmethod
    def is_complete(self) -> bool:
        """Returns :const:`True` if all of the final edits are available.

        Note:
            This should return :const:`True` if the edit can determine that its representation will no longer change,
            regardless of whether our bounds have been fully tightened.

        """
        raise NotImplementedError()

    @property
    @abstractmethod
    def valid(self) -> bool:
        """Returns :const:`False` if the edit has determined that it is no longer valid"""
        raise NotImplementedError()

    @valid.setter
    @abstractmethod
    def valid(self, is_valid: bool):
        raise NotImplementedError()

    def print(self, formatter: GraphtageFormatter, printer: Printer):
        """Edits can optionally implement a printing method

        This function is called automatically from the formatter in the
        :ref:`Printing Protocol` and should never be called directly unless you really know what you're doing!
        Raising :exc:`NotImplementedError` will cause the formatter to fall back on its own printing implementations.

        """
        raise NotImplementedError()

    def on_diff(self, from_node: Union['EditedTreeNode', 'TreeNode']):
        """A callback for when an edit is assigned to an :class:`EditedTreeNode` in :meth:`TreeNode.diff`.

        This default implementation adds the edit to the node::

            from_node.edit = self
            from_node.edit_list.append(self)

        Implementations of the :class:`Edit` protocol that have sub-edits (like :class:`CompoundEdit`) should
        recursively call :meth:`Edit.on_diff` on its sub-edits.

        Args:
            from_node: The edited node that was added to the diff

        """

        from_node.edit = self
        from_node.edit_list.append(self)


@runtime_checkable
class CompoundEdit(Edit, Iterable, Protocol):
    """A protocol for edits that are composed of sub-edits"""

    @abstractmethod
    def edits(self) -> Iterator[Edit]:
        """Returns an iterator over this edit's sub-edits"""
        raise NotImplementedError()

    def on_diff(self, from_node: 'EditedTreeNode'):
        """A callback for when an edit is assigned to an :class:`EditedTreeNode` in :meth:`TreeNode.diff`.

        This default implementation adds the edit to the node, and recursively calls :meth:`Edit.on_diff` on all of
        the sub-edits::

            from_node.edit = self
            from_node.edit_list.append(self)
            for edit in self.edits():
                edit.on_diff(edit.from_node)

        Args:
            from_node: The edited node that was added to the diff

        """
        if hasattr(from_node, 'edit_list'):
            from_node.edit_list.append(self)
        if hasattr(from_node, 'edit'):
            from_node.edit = self
        for edit in self.edits():
            edit.on_diff(edit.from_node)


def explode_edits(edit: Edit) -> Iterator[Edit]:
    """Performs a depth-first traversal over a potentially compound edit.

    If an edit implements the :class:`CompoundEdit` protocol, its sub-edits are recursively included in the output.

    This implementation is equivalent to::

        if isinstance(edit, CompoundEdit):
            return itertools.chain(*map(explode_edits, edit.edits()))
        else:
            return iter((edit,))

    Args:
        edit: The edit that is to be exploded

    Returns:
        Iterator[Edit]: An iterator over the edits.

    """
    if isinstance(edit, CompoundEdit):
        return itertools.chain(*map(explode_edits, edit.edits()))
    else:
        return iter((edit,))


E = TypeVar('E', bound=Union['EditedTreeNode', 'TreeNode'])
T = TypeVar('T', bound='TreeNode')


class EditedTreeNode:
    """A mixin for a :class:`TreeNode` that has been edited.

    In practice, an object that is an instance of :class:`EditedTreeNode` will always *also* be an instance of
    :class:`TreeNode`.

    This class should almost never be instantiated directly; it is used by :meth:`TreeNode.diff`.

    """
    def __init__(self):
        self.removed: bool = False
        self.inserted: List[TreeNode] = []
        self.matched_to: Optional[TreeNode] = None
        self.edit_list: List[Edit] = []
        self.edit: Optional[Edit] = None

    @property
    def edited(self) -> bool:
        """Edited tree nodes are always edited"""
        return True

    def edited_cost(self) -> int:
        """The cost of the edit applied to this node.

        This will first fully tighten all of the bounds of :obj:`self.edit_list`, and then return the sum of their
        upper bounds::

            while any(e.tighten_bounds() for e in self.edit_list):
                pass
            return sum(e.bounds().upper_bound for e in self.edit_list)

        Since all of the edits are fully tightened, this function returns a :class:`int` instead of a
        :class:`graphtage.bounds.Bounds`.

        Returns:
            int: The sum of the costs of the edits applied to this node.

        """
        while any(e.tighten_bounds() for e in self.edit_list):
            pass
        return sum(e.bounds().upper_bound for e in self.edit_list)


class TreeNode(metaclass=ABCMeta):
    """Abstract base class for nodes in Graphtage's intermediate representation.

    Tree nodes are intended to be immutable. :class:`EditedTreeNode`, on the other hand, can be mutable. See
    :meth:`TreeNode.make_edited`.

    """
    _total_size = None
    _edited_type: Optional[Type[Union[EditedTreeNode, T]]] = None
    edit_modifiers: Optional[List[Callable[['TreeNode', 'TreeNode'], Optional[Edit]]]] = None

    @property
    def edited(self) -> bool:
        """Returns whether this node has been edited.

        The default implementation returns :const:`False`, whereas :meth:`EditedTreeNode.edited` returns :const:`True`.

        """
        return False

    def _edits_with_modifiers(self, node: 'TreeNode') -> Edit:
        for modifier in self.edit_modifiers:
            ret = modifier(self, node)
            if ret is not None:
                return ret
        return self.__class__.edits(self, node)

    def __getattribute__(self, item):
        if item == 'edits' and super().__getattribute__('edit_modifiers'):
            return super().__getattribute__('_edits_with_modifiers')
        else:
            return super().__getattribute__(item)

    @abstractmethod
    def to_obj(self):
        """Returns a pure Python representation of this node.

        For example, a node representing a list, like :class:`graphtage.ListNode`, should return a Python :class:`list`.
        A node representing a mapping, like :class:`graphtage.MappingNode`, should return a Python :class:`dict`.
        Container nodes should recursively call :meth:`TreeNode.to_obj` on all of their children.

        This is used solely for the providing objects to operate on in the commandline expressions evaluation, for
        options like `--match-if` and `--match-unless`.

        """
        raise NotImplementedError()

    @abstractmethod
    def children(self) -> Collection['TreeNode']:
        """Returns a collection of this node's children, if any."""
        raise NotImplementedError()

    def dfs(self) -> Iterator['TreeNode']:
        """Performs a depth-first traversal over all of this node's descendants.

        :obj:`self` is always included and yielded first.

        This implementation is equivalent to::

            stack = [self]
            while stack:
                node = stack.pop()
                yield node
                stack.extend(reversed(node.children()))

        """
        stack = [self]
        while stack:
            node = stack.pop()
            yield node
            stack.extend(reversed(node.children()))

    @property
    def is_leaf(self) -> bool:
        """Returns whether this node is a leaf.

        This implementation is equivalent to::

            return len(self.children()) == 0

        """
        return len(self.children()) == 0

    @abstractmethod
    def edits(self, node: 'TreeNode') -> Edit:
        """Calculates the best edit to transform this node into the provided node.

        Args:
            node: The node to which to transform.

        Returns:
            Edit: The best possible edit.

        """
        raise NotImplementedError()

    @classmethod
    def edited_type(cls) -> Type[Union[EditedTreeNode, T]]:
        """Dynamically constructs a new class that is *both* a :class:`TreeNode` *and* an :class:`EditedTreeNode`.

        The edited type's member variables are populated by the result of :meth:`TreeNode.editable_dict` of the
        :class:`TreeNode` it wraps::

            new_node.__dict__ = dict(wrapped_tree_node.editable_dict())

        Returns:
            Type[Union[EditedTreeNode, T]]: A class that is *both* a :class:`TreeNode` *and* an :class:`EditedTreeNode`.
            Its constructor accepts a :class:`TreeNode` that it will wrap.

        """
        if cls._edited_type is None:
            def init(etn, wrapped_tree_node: TreeNode):
                etn.__dict__ = dict(wrapped_tree_node.editable_dict())
                EditedTreeNode.__init__(etn)

            cls._edited_type = type(f'Edited{cls.__name__}', (EditedTreeNode, cls), {
                '__init__': init
            })
        return cls._edited_type

    def make_edited(self) -> Union[EditedTreeNode, T]:
        """Returns a new, copied instance of this node that is also an instance of :class:`EditedTreeNode`.

        This is equivalent to::

            return self.edited_type()(self)

        Returns:
            Union[EditedTreeNode, T]: A copied version of this node that is also an instance of :class:`EditedTreeNode`
            and thereby mutable.

        """
        ret = self.edited_type()(self)
        assert isinstance(ret, self.__class__)
        assert isinstance(ret, EditedTreeNode)
        return ret

    def editable_dict(self) -> Dict[str, Any]:
        """Copies :obj:`self.__dict__`, calling :meth:`TreeNode.editable_dict` on any :class:`TreeNode` objects therein.

        This is equivalent to::

            ret = dict(self.__dict__)
            if not self.is_leaf:
                for key, value in ret.items():
                    if isinstance(value, TreeNode):
                        ret[key] = value.make_edited()
            return ret

        This is used by :meth:`TreeNode.make_edited`.

        """
        ret = dict(self.__dict__)
        if not self.is_leaf:
            # Deep-copy any sub-nodes
            for key, value in ret.items():
                if isinstance(value, TreeNode):
                    ret[key] = value.make_edited()
        return ret

    def get_all_edits(self, node: 'TreeNode') -> Iterator[Edit]:
        """Returns an iterator over all edits that will transform this node into the provided node.

        Args:
            node: The node to which to transform this one.

        Returns:
            Iterator[Edit]: An iterator over edits. Note that this iterator will automatically
            :func:`explode <explode_edits>` any :class:`CompoundEdit` in the sequence.

        """

        edit = self.edits(node)
        prev_bounds = edit.bounds()
        total_range = prev_bounds.upper_bound - prev_bounds.lower_bound
        prev_range = total_range
        with DEFAULT_PRINTER.tqdm(leave=False, initial=0, total=total_range, desc='Diffing') as t:
            while edit.valid and not edit.is_complete() and edit.tighten_bounds():
                new_bounds = edit.bounds()
                new_range = new_bounds.upper_bound - new_bounds.lower_bound
                t.update(prev_range - new_range)
                prev_range = new_range
        edit_stack = [edit]
        while edit_stack:
            edit = edit_stack.pop()
            if isinstance(edit, CompoundEdit):
                edit_stack.extend(list(edit.edits()))
            else:
                while edit.bounds().lower_bound == 0 and not edit.bounds().definitive() and edit.tighten_bounds():
                    pass
                if edit.bounds().lower_bound > 0:
                    yield edit

    def diff(self: T, node: 'TreeNode') -> Union[EditedTreeNode, T]:
        """Performs a diff against the provided node.

        Args:
            node: The node against which to perform the diff.

        Returns:
            Union[EditedTreeNode, T]: An edited version of this node with all edits being
            :meth:`completed <Edit.is_complete>`.

        """
        ret = self.make_edited()
        assert isinstance(ret, self.__class__)
        assert isinstance(ret, EditedTreeNode)
        edit = ret.edits(node)
        prev_bounds = edit.bounds()
        total_range = prev_bounds.upper_bound - prev_bounds.lower_bound
        prev_range = total_range
        with DEFAULT_PRINTER.tqdm(leave=False, initial=0, total=total_range, desc='Diffing') as t:
            while edit.valid and not edit.is_complete() and edit.tighten_bounds():
                new_bounds = edit.bounds()
                new_range = new_bounds.upper_bound - new_bounds.lower_bound
                if prev_range < new_range:
                    log.warning(f"The most recent call to `tighten_bounds()` on edit {edit} tightened its bounds from {prev_bounds} to {new_bounds}")
                t.update(prev_range - new_range)
                prev_range = new_range
        edit.on_diff(ret)
        return ret

    @property
    def total_size(self) -> int:
        """The size of this node.

        This is an arbitrary, immutable value that is used to calculate the bounded costs of edits on this node.

        The first time this property is called, its value will be set and memoized by calling
        :meth:`TreeNode.calculate_total_size`.

        Returns:
            int: An arbitrary integer representing the size of this node.

        """
        if self._total_size is None:
            self._total_size = self.calculate_total_size()
        return self._total_size

    @abstractmethod
    def calculate_total_size(self) -> int:
        """Calculates the size of this node.
        This is an arbitrary, immutable value that is used to calculate the bounded costs of edits on this node.

        Returns:
            int: An arbitrary integer representing the size of this node.

        """
        return 0

    @abstractmethod
    def print(self, printer: Printer):
        """Prints this node."""
        raise NotImplementedError()


class ContainerNode(TreeNode, Iterable, Sized, ABC):
    """A tree node that has children."""

    def children(self) -> List[TreeNode]:
        """The children of this node.

        Equivalent to::

            list(self)

        """
        return list(self)

    @property
    def is_leaf(self) -> bool:
        """Container nodes are never leaves, even if they have no children.

        Returns:
            bool: :const:`False`

        """
        return False

    def all_children_are_leaves(self) -> bool:
        """Tests whether all of the children of this container are leaves.

        Equivalent to::

            all(c.is_leaf for c in self)

        Returns:
            bool: :const:`True` if all children are leaves.

        """
        return all(c.is_leaf for c in self)
