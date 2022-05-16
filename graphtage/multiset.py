"""A module for representing an edit on a multiset.

This is used by :class:`graphtage.MultiSetNode` and :class:`graphtage.DictNode`, since the latter is a multiset containg
:class:`graphtage.KeyValuePairNode` objects.

"""

from typing import Iterator, List

import graphtage
from .bounds import Range
from .edits import Insert, Match, Remove
from .matching import WeightedBipartiteMatcher
from .sequences import SequenceEdit, SequenceNode
from .tree import Edit, TreeNode
from .utils import HashableCounter, largest


class MultiSetEdit(SequenceEdit):
    """An edit matching one unordered collection of items to another.

    It works by using a :class:`graphtage.matching.WeightedBipartiteMatcher` to find the minimum cost matching from
    the elements of one collection to the elements of the other.

    """
    def __init__(
            self,
            from_node: SequenceNode,
            to_node: SequenceNode,
            from_set: HashableCounter[TreeNode],
            to_set: HashableCounter[TreeNode],
            auto_match_keys: bool = True
    ):
        """Initializes the edit.

        Args:
            from_node: Any sequence node from which to match.
            to_node: Any sequence node to which to match.
            from_set: The set of nodes from which to match. These should typically be children of :obj:`from_node`, but
                this is neither checked nor enforced.
            to_set: The set of nodes to which to match. These should typically be children of :obj:`to_node`, but this
                is neither checked nor enforced.
            auto_match_keys: If `True`, any :class:`graphtage.KeyValuePairNode`s in :obj:`from_set` that have keys
                equal to :class:`graphtage.KeyValuePairNode`s in :obj:`to_set` will automatically be matched. Setting
                this to `False` will require a significant amount more computation for larger dictionaries.

        """
        self._matched_kvp_edits: List[Edit] = []
        if auto_match_keys:
            to_set = HashableCounter(to_set)
            from_set = HashableCounter(from_set)
            to_remove_from = []
            for f in from_set.keys():
                if not isinstance(f, graphtage.KeyValuePairNode):
                    continue
                for t in to_set.keys():
                    if not isinstance(f, graphtage.KeyValuePairNode):
                        continue
                    if f.key == t.key:
                        num_matched = min(from_set[f], to_set[t])
                        for _ in range(num_matched):
                            self._matched_kvp_edits.append(f.edits(t))
                        to_remove_from.append((f, num_matched))
                        break
                else:
                    continue
                to_set[t] -= num_matched
            for f, num_matched in to_remove_from:
                from_set[f] -= num_matched
        self.to_insert = to_set - from_set
        """The set of nodes in :obj:`to_set` that do not exist in :obj:`from_set`."""
        self.to_remove = from_set - to_set
        """The set of nodes in :obj:`from_set` that do not exist in :obj:`to_set`."""
        to_match = from_set & to_set
        self._edits: List[Edit] = [Match(n, n, 0) for n in to_match.elements()]
        self._matcher = WeightedBipartiteMatcher(
            from_nodes=self.to_remove.elements(),
            to_nodes=self.to_insert.elements(),
            get_edge=lambda f, t: f.edits(t)
        )
        super().__init__(
            from_node=from_node,
            to_node=to_node
        )

    def is_complete(self) -> bool:
        return self._matcher.is_complete()

    def edits(self) -> Iterator[Edit]:
        yield from self._edits
        yield from self._matched_kvp_edits
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

    def tighten_bounds(self) -> bool:
        """Delegates to :meth:`WeightedBipartiteMatcher.tighten_bounds`."""
        for kvp_edit in self._matched_kvp_edits:
            if kvp_edit.tighten_bounds():
                return True
        return self._matcher.tighten_bounds()

    def bounds(self) -> Range:
        b = self._matcher.bounds()
        for kvp_edit in self._matched_kvp_edits:
            b = b + kvp_edit.bounds()
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
