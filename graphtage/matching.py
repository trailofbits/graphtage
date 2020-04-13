import itertools
from abc import ABCMeta, abstractmethod
from collections.abc import Set as SetCollection
from typing import Callable, Dict, Generic, Iterable, Iterator, List
from typing import Mapping, Optional, Sequence, Set, Tuple, TypeVar, Union

import numpy as np
from scipy.optimize import linear_sum_assignment

from .bounds import Bounded, Range
from .bounds import sort as bounds_sort
from .fibonacci import FibonacciHeap


T = TypeVar('T')


class Edge(Generic[T], Bounded):
    def __init__(self, from_node: 'MatchingFromNode[T]', to_node: 'MatchingToNode[T]', weight: Bounded):
        self.from_node: MatchingFromNode[T] = from_node
        self.to_node: MatchingToNode[T] = to_node
        self.weight: Bounded = weight

    def bounds(self) -> Range:
        return self.weight.bounds()

    def tighten_bounds(self) -> bool:
        return self.weight.tighten_bounds()

    @property
    def cost_star(self) -> int:
        while not self.weight.bounds().definitive() and self.weight.tighten_bounds():
            pass
        assert self.weight.bounds().definitive()
        return self.weight.bounds().upper_bound +  self.from_node.potential + max(
            max(n.potential for n in self.from_node.matcher.from_nodes),
            max(n.potential for n in self.to_node.matcher.to_nodes)
        )

    @property
    def cost_bar(self) -> int:
        while not self.weight.bounds().definitive() and self.weight.tighten_bounds():
            pass
        assert self.weight.bounds().definitive()
        return self.weight.bounds().upper_bound + self.from_node.potential - self.to_node.potential

    def __hash__(self):
        return hash((self.from_node, self.to_node))

    def __repr__(self):
        return f"{self.__class__.__name__}(from_node={self.from_node!r}, to_node={self.to_node!r}, weight={self.weight!r})"


class MatchingNode(Generic[T], metaclass=ABCMeta):
    def __init__(self, matcher: 'WeightedBipartiteMatcher[T]', node: T):
        self.node = node
        self.matcher = matcher
        self.weight = Range()
        self._edges: Optional[Dict[MatchingNode[T], Edge[T]]] = None
        self.potential: int = 0

    @abstractmethod
    def construct_edges(self) -> Dict['MatchingNode[T]', Edge[T]]:
        pass

    def edges(self) -> Iterable[Edge[T]]:
        if self._edges is None:
            self._edges = self.construct_edges()
        return self._edges.values()

    def __repr__(self):
        return repr(self.node)

    def __getitem__(self, neighbor: 'MatchingNode[T]') -> Edge[T]:
        if self._edges is None:
            self.edges
        return self._edges[neighbor]

    def __contains__(self, node):
        if self._edges is None:
            self.edges
        return node in self._edges

    def __hash__(self):
        return hash(self.node)

    def __eq__(self, other):
        return self is other


class SortedEdges(Generic[T]):
    def __init__(self, edges: Iterable[Edge[T]]):
        self._edges = edges
        self._sorting_iter: Optional[Iterator[Edge]] = None
        self._sorted: List[Edge[T]] = []
        self._indexes: Dict[MatchingToNode[T], int] = {}

    def _get_next(self) -> bool:
        if self._sorting_iter is None:
            self._sorting_iter = bounds_sort(self._edges)
        try:
            edge = next(self._sorting_iter)
            self._indexes[edge.to_node] = len(self._sorted)
            self._sorted.append(edge)
            return True
        except StopIteration:
            return False

    def head(self) -> Edge[T]:
        if not self._sorted and not self._get_next():
            raise IndexError()
        return self._sorted[0]

    def tail(self) -> Edge[T]:
        while self._get_next():
            pass
        return self._sorted[-1]

    def __getitem__(self, node_or_index: Union['MatchingToNode[T]', int]) -> Union[Edge[T], int]:
        if isinstance(node_or_index, int):
            while len(self._sorted) <= node_or_index:
                if not self._get_next():
                    raise IndexError(node_or_index)
            return self._sorted[node_or_index]
        elif node_or_index not in self._indexes:
            raise IndexError(node_or_index)
        else:
            return self._indexes[node_or_index]


class MatchingFromNode(Generic[T], MatchingNode[T]):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._sorted_neighbors: Optional[SortedEdges[T]] = None

    @property
    def sorted_neighbors(self) -> SortedEdges[T]:
        if self._sorted_neighbors is None:
            self._sorted_neighbors = SortedEdges(self.edges())
        return self._sorted_neighbors

    def construct_edges(self) -> Dict[MatchingNode[T], Edge[T]]:
        return {
            neighbor: Edge(self, neighbor, edge) for neighbor, edge in (
                (neighbor, self.matcher.get_edge(self.node, neighbor.node)) for neighbor in self.matcher.to_nodes
            ) if edge is not None
        }

    def __repr__(self):
        return f"{self.node!r}\u21A6"


class MatchingToNode(Generic[T], MatchingNode[T]):
    def construct_edges(self) -> Dict[MatchingNode[T], Edge[T]]:
        return {
            from_node: from_node[self] for from_node in self.matcher.from_nodes if self in from_node
        }

    def __repr__(self):
        return f"\u21A3{self.node!r}"


class Matching(Generic[T], SetCollection, Set[Edge[T]], Bounded):
    def __init__(self):
        super().__init__()
        self._edges: Set[Edge[T]] = set()
        self._edges_by_node: Dict[MatchingNode[T], Edge[T]] = {}

    def __contains__(self, edge_or_node: Union[Edge[T], MatchingNode[T]]) -> bool:
        if isinstance(edge_or_node, Edge):
            return edge_or_node in self._edges
        else:
            return edge_or_node in self._edges_by_node

    def __len__(self) -> int:
        return len(self._edges)

    def __iter__(self) -> Iterator[Edge[T]]:
        return iter(self._edges)

    def __getitem__(self, node: MatchingNode[T]) -> Optional[Edge[T]]:
        return self._edges_by_node.get(node, None)

    def symmetric_difference(self, matching: Set[Edge[T]]) -> 'Matching[T]':
        ret = Matching()
        ret._edges = self._edges.symmetric_difference(matching)
        ret._edges_by_node = {}
        for edge in ret._edges:
            ret._edges_by_node[edge.from_node] = edge
            ret._edges_by_node[edge.to_node] = edge
        return ret

    def add(self, edge: Edge[T]):
        self._edges.add(edge)
        assert edge.from_node not in self._edges_by_node
        assert edge.to_node not in self._edges_by_node
        self._edges_by_node[edge.from_node] = edge
        self._edges_by_node[edge.to_node] = edge

    def tighten_bounds(self) -> bool:
        for edge in self:
            if edge.weight.tighten_bounds():
                return True
        return False

    def bounds(self) -> Range:
        return sum(edge.weight.bounds() for edge in self._edges)

    def __repr__(self):
        matchings = ", ".join((f"{e.from_node}{e.to_node}" for e in self._edges))
        return f'{self.__class__.__name__}<{matchings}>'


class PathSet(Generic[T], Matching[T]):
    """A version of a Matching with edge directions overridden, used for Karp's weighted bipartite matching algorithm"""
    def __init__(self):
        super().__init__()
        self._flipped: Dict[MatchingToNode[T], Edge[T]] = {}

    def add(self, edge: Edge[T], flip_direction: bool):
        if flip_direction:
            self._flipped[edge.to_node] = edge
        else:
            super().add(edge)

    def _path_to(
            self,
            from_any_of: Set[MatchingFromNode[T]],
            node: MatchingNode[T],
            history: Optional[Set[Edge[T]]] = None
    ) -> Optional[Set[Edge[T]]]:
        if node in from_any_of:
            return set()
        if history is None:
            history = set()
        for source in (self._flipped, self):
            if node in source:
                edge = source[node]
                if edge not in history:
                    history.add(edge)
                    if edge.to_node is node:
                        other_node = edge.from_node
                    else:
                        other_node = edge.to_node
                    ret = self._path_to(from_any_of, other_node, history)
                    if ret is not None:
                        return {edge} | ret
        return None

    def path_to(
        self,
        from_any_of: Set[MatchingFromNode[T]],
        node: MatchingToNode[T]
    ) -> Set[Edge[T]]:
        ret = self._path_to(from_any_of, node)
        if ret is None:
            return set()
        else:
            return ret


class QueueElement:
    def __init__(self, edge: Edge[T], cost: int, is_special: bool):
        self.edge = edge
        self.cost = cost
        self.is_special = is_special

    def __repr__(self):
        return f"<{self.edge!r}, {self.cost!r}{['', ', SPECIAL'][self.is_special]}>"


QueueType = FibonacciHeap[QueueElement, int]


class WeightedBipartiteMatcherPARTIAL_IMPLEMENTATION(Generic[T], Bounded):
    """Partial implementation of AN ALGORITHM TO SOLVE THE mxn ASSIGNMENT PROBLEM IN EXPECTED TIME 0(mn log n)
    by R. M. Karp, 1978
    https://www2.eecs.berkeley.edu/Pubs/TechRpts/1978/ERL-m-78-67.pdf

    The implementation is partial because I realized partway through that, even though this implementation has better
    asymptotic bounds, `scipy.optimize.linear_sum_assignment(...)` will likely be much faster since it is implemented
    in C++ and not pure Python.
    """
    def __init__(
            self,
            from_nodes: Iterable[T],
            to_nodes: Iterable[T],
            get_edge: Callable[[T, T], Optional[Bounded]]
    ):
        self.from_nodes: List[MatchingFromNode[T]] = [MatchingFromNode(self, node) for node in from_nodes]
        self.to_nodes: List[MatchingToNode[T]] = [MatchingToNode(self, node) for node in to_nodes]
        if len(self.from_nodes) > len(self.to_nodes):
            raise ValueError()
        self.get_edge = get_edge
        self.matching: Matching[T] = Matching()
        self._min_path_cost: Dict[MatchingNode[T], int] = {}

    def free_sources(self) -> Iterator[MatchingFromNode[T]]:
        # Given a matching M, a vertex v is called free if it is incident with no edge in M
        for source in self.from_nodes:
            for edge in source.edges():
                if edge in self.matching:
                    break
            else:
                yield source

    def free_destinations(self) -> Iterator[MatchingToNode[T]]:
        # Given a matching M, a vertex v is called free if it is incident with no edge in M
        for destination in self.to_nodes:
            for edge in destination.edges():
                if edge in self.matching:
                    break
            else:
                yield destination

    def _select(self, priority_queue: QueueType) -> Edge[T]:
        while True:
            element = priority_queue.pop()
            x, y = element.edge.from_node, element.edge.to_node
            if element.is_special:
                if x.sorted_neighbors.tail().to_node != y:
                    # Find the successor of x, y in the sorted neighbors list
                    y_index: int = x.sorted_neighbors[y]
                    edge_w: Edge[T] = x.sorted_neighbors[y_index + 1]
                    assert edge_w.from_node == x
                    # TODO: Figure out if this is correct/a typo in the paper
                    priority_queue.push(QueueElement(
                        edge=edge_w,
                        cost=self._min_path_cost[x] + edge_w.cost_star,
                        is_special=True
                    ))
                priority_queue.push(QueueElement(
                    edge=element.edge,
                    cost=self._min_path_cost[x] + element.edge.cost_bar,
                    is_special=False
                ))
            elif element.edge not in self.matching:
                return element.edge

    def tighten_bounds(self) -> bool:
        if len(self.matching) >= len(self.from_nodes):
            return self.matching.tighten_bounds()

        path: PathSet[T] = PathSet()
        q: QueueType = FibonacciHeap(key=lambda n: n.cost)
        r: Set[MatchingNode[T]] = set(self.free_sources())
        for x in r:
            self._min_path_cost[x] = 0
            assert isinstance(x, MatchingFromNode)
            edge = x.sorted_neighbors.head()
            q.push(QueueElement(edge=edge, cost=edge.cost_star, is_special=True))
        while not (set(self.free_destinations()) & r):
            # At this point, we know that r and self.free_destinations() have no common elements
            edge: Edge[T] = self._select(q)
            y = edge.to_node
            if y not in r:
                path.add(edge, flip_direction=False)
                r.add(y)
                self._min_path_cost[y] = edge.from_node.potential + edge.cost_bar
                for e in y.edges():
                    if e in self.matching:
                        y_is_free = False
                        break
                else:
                    y_is_free = True
                if not y_is_free:
                    incident = self.matching[y]
                    assert incident is not None
                    assert y is incident.to_node
                    path.add(incident, flip_direction=True)
                    vee = incident.from_node
                    r.add(vee)
                    self._min_path_cost[vee] = self._min_path_cost[y]
                    ell = next(incident.from_node.neighbors_by_weight)
                    q.push(QueueElement(edge=ell, cost=self._min_path_cost[vee] + ell.cost_star, is_special=True))
            for v in self.from_nodes:
                if v not in r:
                    self._min_path_cost[v] = self._min_path_cost[edge.to_node]
                v.potential += self._min_path_cost[v]
            for v in self.to_nodes:
                if v not in r:
                    self._min_path_cost[v] = self._min_path_cost[edge.to_node]
                v.potential += self._min_path_cost[v]
            # Let ~P be the unique directed path from a free source to y whose edges are all in PATHSET;
            tilde_p = path.path_to(set(self.free_sources()), edge.to_node)
            # Let P be the set of edges in G corresponding to directed edges in P;
            # M <- M©P
            self.matching = self.matching.symmetric_difference(tilde_p)
            print(self.matching)

    def bounds(self) -> Range:
        pass


W = TypeVar('W', bound=Union[bool, int, float])
EdgeType = Union[bool, int, float]

INTEGER_DTYPE_INTERVALS: Tuple[Tuple[int, int, np.dtype], ...] = (
    (0, 2**8, np.dtype(np.uint8)),
    (0, 2**16, np.dtype(np.uint16)),
    (0, 2**32, np.dtype(np.uint32)),
    (0, 2**64, np.dtype(np.uint64)),
    (-2**7, 2**7, np.dtype(np.int8)),
    (-2**15, 2**15, np.dtype(np.int16)),
    (-2**31, 2**31, np.dtype(np.int32)),
    (-2**63, 2**63, np.dtype(np.int64))
)


def get_dtype(min_value: int, max_value: int) -> np.dtype:
    """Returns the smallest numpy dtype capable of storing integers in the range [min_value, max_value]"""
    for min_range, max_range, dtype in INTEGER_DTYPE_INTERVALS:
        if min_range <= min_value and max_range > max_value:
            return dtype
    return np.dtype(int)


def min_weight_bipartite_matching(
        from_nodes: Sequence[T],
        to_nodes: Sequence[T],
        get_edges: Callable[[T, T], Optional[W]]
) -> Mapping[int, Tuple[int, Optional[EdgeType]]]:
    # Assume that the bipartite graph is dense. If the edges are sparse, consider switching to `scipy.sparse.coo_matrix`
    weights: List[List[Optional[EdgeType]]] = [[None] * len(to_nodes) for _ in range(len(from_nodes))]

    edge_type: Optional[type] = None
    max_edge: Optional[EdgeType] = None
    min_edge: Optional[EdgeType] = None

    has_null_edges = False

    for (from_index, from_node), (to_index, to_node) in itertools.product(enumerate(from_nodes), enumerate(to_nodes)):
        edge = get_edges(from_node, to_node)
        if edge is not None:
            if edge_type is None:
                edge_type = type(edge)
            elif edge_type is not type(edge):
                raise ValueError(f"The edge between {from_node!r} and {to_node!r} was expected to be of type {edge_type} but instead receieved {type(edge)}")
            if max_edge is None or max_edge < edge:
                max_edge = edge
            if min_edge is None or min_edge > edge:
                min_edge = edge
            weights[from_index][to_index] = edge
        else:
            has_null_edges = True

    if has_null_edges:
        if isinstance(edge_type, bool):
            raise ValueError("Null edges are only supported with `int` or `float` edge types, not `bool`. Bipartite graphs with `bool` edge weights must be complete.")
        null_edge_value: Optional[EdgeType] = max(
            sum(weights[row][col] for row in range(len(from_nodes)) if weights[row][col] is not None)
            for col in range(len(to_nodes))
        ) + 1
        assert null_edge_value > max_edge
        max_edge = null_edge_value
    else:
        null_edge_value = None

    if edge_type is None:
        # There are no edges in the graph
        return {}
    elif edge_type is bool:
        dtype = bool
    elif edge_type is float:
        dtype = float
    elif not edge_type is int:
        raise ValueError(f"Unexpected edge type: {edge_type}")
    else:
        dtype = get_dtype(min_edge, max_edge)

    if has_null_edges:
        for row in range(len(from_nodes)):
            for col in range(len(to_nodes)):
                if weights[row][col] is None:
                    weights[row][col] = null_edge_value

    left_matches = linear_sum_assignment(np.array(weights, dtype=dtype), maximize=False)
    return {
        from_index: (to_index, weights[from_index][to_index])
        for from_index, to_index in zip(*left_matches)
        if not has_null_edges or weights[from_index][to_index] < null_edge_value
    }


class WeightedBipartiteMatcher(Generic[T], Bounded):
    def __init__(
            self,
            from_nodes: Iterable[T],
            to_nodes: Iterable[T],
            get_edge: Callable[[T, T], Optional[Bounded]]
    ):
        pass