"""A module for representing an edit on a multiset.

This is used by :class:`graphtage.MultiSetNode` and :class:`graphtage.DictNode`, since the latter is a multiset containg
:class:`graphtage.KeyValuePairNode` objects.

"""

import logging
from collections.abc import Iterator, Sequence

import graphtage

from . import batch_distance
from .batch_distance import Pair
from .bounds import Range
from .edits import Insert, Match, Remove
from .matching import WeightedBipartiteMatcher
from .sequences import SequenceEdit, SequenceNode
from .tree import Edit, TreeNode
from .utils import HashableCounter, largest

log = logging.getLogger(__name__)

MATCHING_SIZE_WARNING_THRESHOLD = 250_000
"""The number of candidate pairs above which matching two unordered collections is reported as slow.

The pairs are priced in one batch by :mod:`graphtage.batch_distance`, so this counts the pairs a diff can price in
roughly a second on an Apple M-series laptop rather than the few hundred that were worth warning about when each
pair cost its own pure Python dynamic program.

"""


def _first_with_key(to_set: HashableCounter[TreeNode], key: TreeNode) -> TreeNode | None:
    """Returns the first key/value pair in a multiset whose key equals the one given.

    Args:
        to_set: The multiset to search.
        key: The key to look for.

    Returns:
        Optional[TreeNode]: The matching key/value pair, or :const:`None` if the multiset holds none.

    """
    for t in to_set:
        if isinstance(t, graphtage.KeyValuePairNode) and t.key == key:
            return t
    return None


def _match_by_key(
        from_set: HashableCounter[TreeNode],
        to_set: HashableCounter[TreeNode],
) -> list[tuple[TreeNode, TreeNode, int]]:
    """Pairs off key/value pairs whose keys are equal, and takes what it pairs out of both multisets.

    This decides the pairing without constructing any edit for it, so that the caller can price every string the
    resulting edits will ask about in one batch before it builds the first of them.

    Args:
        from_set: The multiset to match from. Its counts are decremented in place.
        to_set: The multiset to match to. Its counts are decremented in place.

    Returns:
        List[Tuple[TreeNode, TreeNode, int]]: Each matched pair and the number of times it matched.

    """
    matched: list[tuple[TreeNode, TreeNode, int]] = []
    for f in from_set:
        if not isinstance(f, graphtage.KeyValuePairNode):
            continue
        t = _first_with_key(to_set, f.key)
        if t is None:
            continue
        num_matched = min(from_set[f], to_set[t])
        matched.append((f, t, num_matched))
        to_set[t] -= num_matched
    for f, _, num_matched in matched:
        from_set[f] -= num_matched
    return matched


def _leaf_pair(from_node: TreeNode, to_node: TreeNode) -> Pair | None:
    """Returns the pair of strings that costing an edit between two leaves compares.

    Args:
        from_node: The leaf to match from.
        to_node: The leaf to match to.

    Returns:
        Optional[Pair]: The pair, or :const:`None` when one side is a :class:`str` and the other :class:`bytes`,
        which a batch will not price.

    """
    if isinstance(from_node, graphtage.StringNode) and isinstance(to_node, graphtage.StringNode):
        if isinstance(from_node.object, str) == isinstance(to_node.object, str):
            return from_node.object, to_node.object
        return None
    return str(from_node.object), str(to_node.object)


def _collect_pairs(from_node: TreeNode, to_node: TreeNode, pairs: list[Pair]) -> None:
    """Appends the string pairs that costing an edit between two nodes will ask about.

    Two key/value pairs contribute their keys and their values, because
    :class:`graphtage.KeyValuePairEdit` always matches key to key and value to value. Two containers contribute
    nothing: the edit between them prices its own cross product when it is constructed.

    Args:
        from_node: The node to match from.
        to_node: The node to match to.
        pairs: The list to append to.

    """
    if isinstance(from_node, graphtage.LeafNode) and isinstance(to_node, graphtage.LeafNode):
        pair = _leaf_pair(from_node, to_node)
        if pair is not None:
            pairs.append(pair)
    elif isinstance(from_node, graphtage.KeyValuePairNode) and isinstance(to_node, graphtage.KeyValuePairNode):
        _collect_pairs(from_node.key, to_node.key, pairs)
        _collect_pairs(from_node.value, to_node.value, pairs)


def _preprice_matching(
        matched: Sequence[tuple[TreeNode, TreeNode, int]],
        to_remove: HashableCounter[TreeNode],
        to_insert: HashableCounter[TreeNode],
        num_pairs: int,
) -> None:
    """Prices in one batch every string pair that costing a multiset matching will ask about.

    Args:
        matched: The key/value pairs that :func:`_match_by_key` paired off.
        to_remove: The nodes the matcher will match from.
        to_insert: The nodes the matcher will match to.
        num_pairs: The number of candidate pairs the matcher faces, used to decide whether a batch is worth it.

    """
    if num_pairs + len(matched) < batch_distance.PREPRICE_MIN_PAIRS:
        return
    pairs: list[Pair] = []
    for f, t, _ in matched:
        _collect_pairs(f, t, pairs)
    for f in to_remove:
        for t in to_insert:
            _collect_pairs(f, t, pairs)
    if pairs:
        batch_distance.preprice(pairs)


class MultiSetEdit(SequenceEdit):
    """An edit matching one unordered collection of items to another.

    It works by using a :class:`graphtage.matching.WeightedBipartiteMatcher` to find the minimum cost matching from
    the elements of one collection to the elements of the other.

    """

    __slots__ = ('_edits', '_matched_kvp_edits', '_matcher', 'to_insert', 'to_remove')

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
        matched_key_pairs: list[tuple[TreeNode, TreeNode, int]] = []
        if auto_match_keys:
            to_set = HashableCounter(to_set)
            from_set = HashableCounter(from_set)
            matched_key_pairs = _match_by_key(from_set, to_set)
        self.to_insert = to_set - from_set
        """The set of nodes in :obj:`to_set` that do not exist in :obj:`from_set`."""
        self.to_remove = from_set - to_set
        """The set of nodes in :obj:`from_set` that do not exist in :obj:`to_set`."""
        to_match = from_set & to_set
        num_pairs = sum(self.to_remove.values()) * sum(self.to_insert.values())
        if num_pairs > MATCHING_SIZE_WARNING_THRESHOLD:
            log.warning(
                "Matching %d unordered elements against %d requires costing %d pairs, which can take a long time",
                sum(self.to_remove.values()), sum(self.to_insert.values()), num_pairs
            )
        _preprice_matching(matched_key_pairs, self.to_remove, self.to_insert, num_pairs)
        self._matched_kvp_edits: list[Edit] = [
            f.edits(t) for f, t, num_matched in matched_key_pairs for _ in range(num_matched)
        ]
        self._edits: list[Edit] = [Match(n, n, 0) for n in to_match.elements()]
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
