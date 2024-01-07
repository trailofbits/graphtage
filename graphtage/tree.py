import itertools
import logging
import sys
from abc import abstractmethod, ABC, ABCMeta
from functools import wraps
from typing import (
    Any, Callable, Dict, Iterable, Iterator, List, Optional, Sequence, Sized, Tuple, Type, TypeVar, Union
)
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
class CompoundEdit(Edit, Protocol):
    """A protocol for edits that are composed of sub-edits"""

    @abstractmethod
    def __iter__(self) -> Iterator[Edit]:
        """Returns an iterator over this edit's sub-edits."""
        raise NotImplementedError()

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


class TreeNodeMeta(ABCMeta):
    def __init__(cls, name, *args, **kwargs):
        super().__init__(name, *args, **kwargs)
        cls._edited_type: Optional[Type[Union[EditedTreeNode, T]]] = None

    def edited_type(self) -> Type[Union[EditedTreeNode, T]]:
        """Dynamically constructs a new class that is *both* a :class:`TreeNode` *and* an :class:`EditedTreeNode`.

        The edited type's member variables are populated by the result of :meth:`TreeNode.editable_dict` of the
        :class:`TreeNode` it wraps::

            new_node.__dict__ = dict(wrapped_tree_node.editable_dict())

        Returns:
            Type[Union[EditedTreeNode, T]]: A class that is *both* a :class:`TreeNode` *and* an :class:`EditedTreeNode`.
            Its constructor accepts a :class:`TreeNode` that it will wrap.

        """
        if self._edited_type is None:
            if issubclass(self, EditedTreeNode):
                return self

            def init(etn, wrapped_tree_node: TreeNode):
                parent_before = wrapped_tree_node.parent
                try:
                    wrapped_tree_node._parent = None
                    etn.__dict__ = {k: v for k, v in wrapped_tree_node.editable_dict().items() if k != "_parent"}
                finally:
                    wrapped_tree_node._parent = parent_before
                EditedTreeNode.__init__(etn)

            def edited_copy(etn):
                return etn.__class__(etn)

            self._edited_type = type(f'Edited{self.__name__}', (EditedTreeNode, self), {
                '__init__': init,
                'copy': edited_copy
            })
        return self._edited_type


class TreeNode(metaclass=TreeNodeMeta):
    """Abstract base class for nodes in Graphtage's intermediate representation.

    Tree nodes are intended to be immutable. :class:`EditedTreeNode`, on the other hand, can be mutable. See
    :meth:`TreeNode.make_edited`.

    """
    _total_size = None
    _parent: Optional["TreeNode"] = None
    _edit_modifiers: Optional[List[Callable[["TreeNode", "TreeNode"], Optional[Edit]]]] = None

    @property
    def edited(self) -> bool:
        """Returns whether this node has been edited.

        The default implementation returns :const:`False`, whereas :meth:`EditedTreeNode.edited` returns :const:`True`.

        """
        return False

    def _edits_with_modifiers(self, node: 'TreeNode') -> Edit:
        for modifier in self._edit_modifiers:
            ret = modifier(self, node)
            if ret is not None:
                return ret
        return self.__class__.edits(self, node)

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

    def copy_from(self: T, children: Iterable["TreeNode"]) -> T:
        """Constructs a new instance of this tree node from a list of its children"""
        return self.__class__(*children)

    def copy(self: T) -> T:
        """Creates a deep copy of this node"""
        work: List[Tuple[TreeNode, List[TreeNode], List[TreeNode]]] = [(self, [], list(reversed(self.children())))]
        while work:
            node, processed_children, remaining_children = work.pop()
            if not remaining_children:
                new_node = node.copy_from(processed_children)
                if not work:
                    return new_node
                work[-1][1].append(new_node)
            else:
                child = remaining_children.pop()
                work.append((node, processed_children, remaining_children))
                work.append((child, [], list(reversed(child.children()))))
        raise NotImplementedError("This should not be reachable")

    @property
    def parent(self) -> Optional["TreeNode"]:
        """The parent node of this node, or :const:`None` if it has no parent.

        The setter for this property should only be called by the parent node setting itself as the parent of its child.

        :class:`ContainerNode` subclasses automatically set this property for all of their children. However, if
        you define a subclass of :class:`TreeNode` *does not* extend off of :class:`ContainerNode` and for which
        ``len(self.children()) > 0``, then each child's parent must be set.

        """
        return self._parent

    @parent.setter
    def parent(self, parent_node: "TreeNode"):
        """This setter should only be called by the parent node setting itself as the parent of its child.

        :class:`ContainerNode` subclasses automatically set this property for all of their children. However, if
        you define a subclass of :class:`TreeNode` *does not* extend off of :class:`ContainerNode` and for which
        ``len(self.children()) > 0``, then each child's parent must be set.

        """
        if self._parent is not None:
            if self._parent is parent_node:
                # we are already assigned to this parent
                return
            raise ValueError(f"Error while setting {self!r}.parent = {parent_node!r}: Parent is already assigned to "
                             f"{self._parent!r}")
        self._parent = parent_node

    @abstractmethod
    def children(self) -> Sequence['TreeNode']:
        """Returns a sequence of this node's children, if any.

        It is the responsibility of any node that has children must set the `.parent` member of each child.
        """
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

    def add_edit_modifier(self, modifier: Callable[["TreeNode", "TreeNode"], Optional[Edit]]):
        if self._edit_modifiers is None:
            self._edit_modifiers = []
            self.edits = self._edits_with_modifiers
        self._edit_modifiers.append(modifier)

    def make_edited(self) -> Union[EditedTreeNode, T]:
        """Returns a new, copied instance of this node that is also an instance of :class:`EditedTreeNode`.

        This is equivalent to::

            return self.__class__.edited_type()(self)

        Returns:
            Union[EditedTreeNode, T]: A copied version of this node that is also an instance of :class:`EditedTreeNode`
            and thereby mutable.

        """
        ret = self.__class__.edited_type()(self)
        if ret._edit_modifiers is not None:
            ret.edits = ret._edits_with_modifiers
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

    def get_all_edit_contexts(self, node: "TreeNode") -> Iterator[Tuple[Tuple["TreeNode", ...], Edit]]:
        """Returns an iterator over all edit contexts that will transform this node into the provided node.

        Args:
            node: The node to which to transform this one.

        Returns:
            Iterator[Tuple[Tuple["TreeNode", ...], Edit]: An iterator over pairs of paths from `node` to the edited
            node, as well as its edit. Note that this iterator will automatically
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
        edit_stack: List[Tuple[Tuple[TreeNode, ...], Edit]] = [((node,), edit)]
        while edit_stack:
            ancestors, edit = edit_stack.pop()
            if isinstance(edit, CompoundEdit):
                for sub_edit in reversed(list(edit.edits())):
                    edit_stack.append((ancestors + (sub_edit.from_node,), sub_edit))
            else:
                while edit.bounds().lower_bound == 0 and not edit.bounds().definitive() and edit.tighten_bounds():
                    pass
                if edit.bounds().lower_bound > 0:
                    yield ancestors, edit

    def get_all_edits(self, node: "TreeNode") -> Iterator[Edit]:
        """Returns an iterator over all edits that will transform this node into the provided node.

        Args:
            node: The node to which to transform this one.

        Returns:
            Iterator[Edit]: An iterator over edits. Note that this iterator will automatically
            :func:`explode <explode_edits>` any :class:`CompoundEdit` in the sequence.

        """
        for _, edit in self.get_all_edit_contexts(node):
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

    def print_parent_context(self, printer: Printer, for_child: "TreeNode"):
        """Prints the context for the given child node.

        For example, if this node represents a list and the child is the element at index 3, then "[3]" might be
        printed.

        The child is expected to be one of this node's children, but this is not validated.

        The default implementation prints nothing.

        """
        pass

    @abstractmethod
    def print(self, printer: Printer):
        """Prints this node."""
        raise NotImplementedError()


class ContainerNode(TreeNode, Sized, ABC):
    """A tree node that has children."""

    @abstractmethod
    def __iter__(self) -> Iterator[TreeNode]:
        raise NotImplementedError()

    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)
        # wrap the subclass's __init__ function to auto-set the parents of its children
        if "__init__" in cls.__dict__ and cls.__init__ is not object.__init__:
            orig_init = getattr(cls, "__init__")

            @wraps(orig_init)
            def wrapped(self: ContainerNode, *args, **kw):
                if hasattr(self, "_container_initializing") and self._container_initializing:
                    first_init = False
                else:
                    setattr(self, "_container_initializing", True)
                    first_init = True
                ret = orig_init(self, *args, **kw)
                if first_init:
                    for child in self.children():
                        child.parent = self
                    if hasattr(self, "_container_initializing"):
                        delattr(self, "_container_initializing")
                return ret

            cls.__init__ = wrapped

    def children(self) -> Sequence[TreeNode]:
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

