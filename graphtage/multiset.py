"""A module for representing an edit on a multiset.

This is used by :class:`graphtage.MultiSetNode` and :class:`graphtage.DictNode`, since the latter is a multiset containg
:class:`graphtage.KeyValuePairNode` objects.

"""

from abc import ABC
from typing import Collection, Generic, Iterable, Iterator, List, Set, TypeVar

from .bounds import Range
from .edits import Insert, Match, Remove
from .matching import WeightedBipartiteMatcher
from .sequences import SequenceEdit, SequenceNode
from .tree import Edit, TreeNode
from .utils import HashableCounter, largest


T = TypeVar("T", bound=Collection[TreeNode])


class AbstractSetEdit(SequenceEdit, Generic[T], ABC):
    """An edit matching one unordered collection of items to another.

        It works by using a :class:`graphtage.matching.WeightedBipartiteMatcher` to find the minimum cost matching from
        the elements of one collection to the elements of the other.

        """

    def __init__(
            self,
            from_node: SequenceNode,
            to_node: SequenceNode,
            from_set: T,
            to_set: T
    ):
        """Initializes the edit.

        Args:
            from_node: Any sequence node from which to match.
            to_node: Any sequence node to which to match.
            from_set: The set of nodes from which to match. These should typically be children of :obj:`from_node`, but
                this is neither checked nor enforced.
            to_set: The set of nodes to which to match. These should typically be children of :obj:`to_node`, but this
                is neither checked nor enforced.

        """
        self.to_insert: T = to_set - from_set
        """The set of nodes in :obj:`to_set` that do not exist in :obj:`from_set`."""
        self.to_remove: T = from_set - to_set
        """The set of nodes in :obj:`from_set` that do not exist in :obj:`to_set`."""
        to_match = from_set & to_set
        self._edits: List[Edit] = [Match(n, n, 0) for n in self.__class__.get_elements(to_match)]
        self._matcher = WeightedBipartiteMatcher(
            from_nodes=self.__class__.get_elements(self.to_remove),
            to_nodes=self.__class__.get_elements(self.to_insert),
            get_edge=lambda f, t: f.edits(t)
        )
        super().__init__(
            from_node=from_node,
            to_node=to_node
        )

    @classmethod
    def get_elements(cls, collection: T) -> Iterable[TreeNode]:
        return collection

    def is_complete(self) -> bool:
        return self._matcher.is_complete()

    def tighten_bounds(self) -> bool:
        """Delegates to :meth:`WeightedBipartiteMatcher.tighten_bounds`."""
        return self._matcher.tighten_bounds()

    def bounds(self) -> Range:
        b = self._matcher.bounds()
        if len(self.to_remove) > len(self.to_insert):
            for edit in largest(
                    *(Remove(to_remove=r, remove_from=self.from_node) for r in self.to_remove),
                    n=len(self.to_remove) - len(self.to_insert),
                    key=lambda e: e.bounds()
            ):
                b = b + edit.bounds()
        elif len(self.to_remove) < len(self.to_insert):
            for edit in largest(
                    *(Insert(to_insert=i, insert_into=self.from_node) for i in self.to_insert),
                    n=len(self.to_insert) - len(self.to_remove),
                    key=lambda e: e.bounds()
            ):
                b = b + edit.bounds()
        return b


class SetEdit(AbstractSetEdit[Set[TreeNode]]):
    def edits(self) -> Iterator[Edit]:
        yield from self._edits
        remove_matched: Set[TreeNode] = set()
        insert_matched: Set[TreeNode] = set()
        for (rem, (ins, edit)) in self._matcher.matching.items():
            yield edit
            remove_matched.add(rem)
            insert_matched.add(ins)
        for rm in (self.to_remove - remove_matched):
            yield Remove(to_remove=rm, remove_from=self.from_node)
        for ins in (self.to_insert - insert_matched):
            yield Insert(to_insert=ins, insert_into=self.from_node)


class MultiSetEdit(AbstractSetEdit[HashableCounter[TreeNode]]):
    @classmethod
    def get_elements(cls, collection: HashableCounter[TreeNode]) -> Iterable[TreeNode]:
        return collection.elements()

    def edits(self) -> Iterator[Edit]:
        yield from self._edits
        remove_matched: HashableCounter[TreeNode] = HashableCounter()
        insert_matched: HashableCounter[TreeNode] = HashableCounter()
        for (rem, (ins, edit)) in self._matcher.matching.items():
            yield edit
            remove_matched[rem] += 1
            insert_matched[ins] += 1
        for rm in (self.to_remove - remove_matched).elements():
            yield Remove(to_remove=rm, remove_from=self.from_node)
        for ins in (self.to_insert - insert_matched).elements():
            yield Insert(to_insert=ins, insert_into=self.from_node)
