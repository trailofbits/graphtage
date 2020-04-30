import itertools
import logging
from abc import abstractmethod, ABC, ABCMeta
from collections.abc import Iterable
from typing import Any, cast, Collection, Dict, Iterator, List, Optional, Sized, Type, TypeVar, Union
from typing_extensions import Protocol, runtime_checkable

from .bounds import Bounded, Range
from .printer import DEFAULT_PRINTER, Printer

log = logging.getLogger(__name__)


class Edit(Bounded, Protocol):
    initial_bounds: Range
    from_node: 'TreeNode'

    @abstractmethod
    def is_complete(self) -> bool:
        """Returns True if all of the final edits() are available,
        regardless of whether our bounds have been fully tightened"""
        raise NotImplementedError()

    @property
    @abstractmethod
    def valid(self) -> bool:
        raise NotImplementedError()

    @valid.setter
    @abstractmethod
    def valid(self, is_valid: bool):
        raise NotImplementedError()

    def print(self, formatter: 'graphtage.formatter.Formatter', printer: Printer):
        raise NotImplementedError()

    def on_diff(self, from_node: 'EditedTreeNode'):
        log.debug(repr(self))
        from_node.edit_list.append(self)


@runtime_checkable
class CompoundEdit(Edit, Iterable, Protocol):
    @abstractmethod
    def edits(self) -> Iterator[Edit]:
        raise NotImplementedError()

    def on_diff(self, from_node: 'EditedTreeNode'):
        log.debug(repr(self))
        if hasattr(from_node, 'edit_list'):
            from_node.edit_list.append(self)
        for edit in self.edits():
            edit.on_diff(edit.from_node)


def explode_edits(edit: Edit) -> Iterator[Edit]:
    if isinstance(edit, CompoundEdit):
        return itertools.chain(*map(explode_edits, edit.edits()))
    else:
        return iter((edit,))


E = TypeVar('E', bound=Union['EditedTreeNode', 'TreeNode'])
T = TypeVar('T', bound='TreeNode')


class EditedTreeNode:
    def __init__(self):
        self.removed: bool = False
        self.inserted: List[TreeNode] = []
        self.matched_to: Optional[TreeNode] = None
        self.edit_list: List[Edit] = []
        self.edit: Optional[Edit] = None

    def edits(self, node: 'TreeNode') -> Edit:
        self.edit = cast(TreeNode, super()).edits(node)
        return self.edit

    @property
    def edited(self) -> bool:
        return True

    def edited_cost(self) -> int:
        while any(e.tighten_bounds() for e in self.edit_list):
            pass
        return sum(e.bounds().upper_bound for e in self.edit_list)


class TreeNode(metaclass=ABCMeta):
    _total_size = None
    _edited_type: Optional[Type[Union[EditedTreeNode, T]]] = None

    @property
    def edited(self) -> bool:
        return False

    @abstractmethod
    def children(self) -> Collection['TreeNode']:
        raise NotImplementedError()

    @property
    def is_leaf(self) -> bool:
        return len(self.children()) == 0

    @abstractmethod
    def edits(self, node: 'TreeNode') -> Edit:
        raise NotImplementedError()

    @classmethod
    def edited_type(cls) -> Type[Union[EditedTreeNode, T]]:
        if cls._edited_type is None:
            def init(etn, wrapped_tree_node: TreeNode):
                etn.__dict__ = dict(wrapped_tree_node.editable_dict())
                EditedTreeNode.__init__(etn)

            cls._edited_type = type(f'Edited{cls.__name__}', (EditedTreeNode, cls), {
                '__init__': init
            })
        return cls._edited_type

    def make_edited(self) -> Union[EditedTreeNode, T]:
        ret = self.edited_type()(self)
        assert isinstance(ret, self.__class__)
        assert isinstance(ret, EditedTreeNode)
        return ret

    def editable_dict(self) -> Dict[str, Any]:
        ret = dict(self.__dict__)
        if not self.is_leaf:
            # Deep-copy any sub-nodes
            for key, value in ret.items():
                if isinstance(value, TreeNode):
                    ret[key] = value.make_edited()
        return ret

    def get_all_edits(self, node: 'TreeNode') -> Iterator[Edit]:
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
                t.update(prev_range - new_range)
                prev_range = new_range
        edit.on_diff(ret)
        return ret

    @property
    def total_size(self) -> int:
        if self._total_size is None:
            self._total_size = self.calculate_total_size()
        return self._total_size

    @abstractmethod
    def calculate_total_size(self) -> int:
        return 0

    @abstractmethod
    def print(self, printer: Printer):
        pass


class ContainerNode(TreeNode, Iterable, Sized, ABC):
    def children(self) -> List[TreeNode]:
        return list(self)

    @property
    def is_leaf(self) -> bool:
        return False

    def all_children_are_leaves(self) -> bool:
        return all(c.is_leaf for c in self)
