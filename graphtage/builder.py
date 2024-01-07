"""A module intended to simplify building Graphtage IR trees from other tree-like data structures."""

from abc import ABC
import logging
from typing import Any, Callable, Dict, Iterable, List, Optional, Tuple, Type, TypeVar

from . import (
    BoolNode, BuildOptions, DictNode, FixedKeyDictNode, FloatNode, IntegerNode, LeafNode, ListNode, MultiSetNode,
    NullNode, StringNode, TreeNode
)
from .object_set import IdentityHash

C = TypeVar("C")
T = TypeVar("T")

log = logging.getLogger(__name__)


class CyclicReference(LeafNode):
    def __init__(self, obj):
        super().__init__(IdentityHash(obj))

    def __hash__(self):
        return id(self.object)

    def __eq__(self, other):
        return isinstance(other, CyclicReference) and other.object is self.object


class Builder(ABC):
    EXPANDERS: Dict[Type[Any], Callable[["Builder", Any], Optional[Iterable[Any]]]]
    BUILDERS: Dict[Type[Any], Callable[["Builder", Any, List[TreeNode]], TreeNode]]

    def __init__(self, options: Optional[BuildOptions] = None):
        if options is None:
            self.options: BuildOptions = BuildOptions()
        else:
            self.options = options

    @staticmethod
    def expander(node_type: Type[T]):
        def wrapper(func: Callable[[C, T], Iterable[Any]]) -> Callable[[C, T], Iterable[Any]]:
            if hasattr(func, "_visitor_expander_for_type"):
                func._visitor_expander_for_type = func._visitor_expander_for_type + (node_type,)
            else:
                setattr(func, "_visitor_expander_for_type", (node_type,))
            return func

        return wrapper

    @staticmethod
    def builder(node_type: Type[T]):
        def wrapper(func: Callable[[C, T, List[TreeNode]], TreeNode]) -> Callable[[C, T, List[TreeNode]], TreeNode]:
            if hasattr(func, "_visitor_builder_for_type"):
                func._visitor_builder_for_type = func._visitor_builder_for_type + (node_type,)
            else:
                setattr(func, "_visitor_builder_for_type", (node_type,))
            return func

        return wrapper

    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)
        if not hasattr(cls, "EXPANDERS") or cls.EXPANDERS is None:
            setattr(cls, "EXPANDERS", {})
        else:
            setattr(cls, "EXPANDERS", dict(cls.EXPANDERS))
        if not hasattr(cls, "BUILDERS") or cls.BUILDERS is None:
            setattr(cls, "BUILDERS", {})
        else:
            setattr(cls, "BUILDERS", dict(cls.BUILDERS))
        new_expanders = {}
        new_builders = {}
        for member_name, member in cls.__dict__.items():
            if hasattr(member, "_visitor_expander_for_type"):
                for expander_type in getattr(member, "_visitor_expander_for_type"):
                    if not isinstance(expander_type, type):
                        raise TypeError(f"{cls.__name__}.{member_name} was registered as an expander for "
                                        f"{expander_type!r}, which is not a type")
                    elif expander_type in cls.EXPANDERS:
                        raise TypeError(f"An expander for type {expander_type.__name__} is already registered to "
                                        f"{cls.EXPANDERS[expander_type]!r} and cannot be re-registered to "
                                        f"{cls.__name__}.{member_name}")
                    elif expander_type in new_expanders:
                        raise TypeError(f"An expander for type {expander_type.__name__} is already registered to "
                                        f"{new_expanders[expander_type]!r} and cannot be re-registered to "
                                        f"{cls.__name__}.{member_name}")
                    new_expanders[expander_type] = member
            if hasattr(member, "_visitor_builder_for_type"):
                for builder_type in getattr(member, "_visitor_builder_for_type"):
                    if not isinstance(builder_type, type):
                        raise TypeError(f"{cls.__name__}.{member_name} was registered as an builder for "
                                        f"{builder_type!r}, which is not a type")
                    elif builder_type in cls.EXPANDERS:
                        raise TypeError(f"A builder for type {builder_type.__name__} is already registered to "
                                        f"{cls.BUILDERS[builder_type]!r} and cannot be re-registered to "
                                        f"{cls.__name__}.{builder_type}")
                    elif builder_type in new_builders:
                        raise TypeError(f"A builder for type {builder_type.__name__} is already registered to "
                                        f"{new_builders[builder_type]!r} and cannot be re-registered to "
                                        f"{cls.__name__}.{builder_type}")
                    new_builders[builder_type] = member
        cls.EXPANDERS.update(new_expanders)
        cls.BUILDERS.update(new_builders)

    def default_expander(self, node: Any) -> Iterable[Any]:
        return ()

    def default_builder(self, node: Any, children: List[TreeNode]) -> TreeNode:
        raise NotImplementedError(f"A builder for type {node.__class__.__name__} is not defined for object {node!r}")

    @classmethod
    def _resolve(cls, obj_type: Type[Any], choices: Dict[Type[Any], T]) -> Optional[T]:
        """Resolves the most specialized expander or builder for `obj_type`"""
        for t in obj_type.__mro__:
            if t in choices:
                return choices[t]
        return None

    @classmethod
    def resolve_expander(cls, obj_type: Type[Any]) -> Optional[Callable[[Any], Optional[Iterable[Any]]]]:
        """Resolves the most specialized expander for `obj_type`"""
        return cls._resolve(obj_type, cls.EXPANDERS)

    @classmethod
    def resolve_builder(cls, obj_type: Type[Any]) -> Optional[Callable[[Any, List[TreeNode]], TreeNode]]:
        """Resolves the most specialized builder for `obj_type`"""
        return cls._resolve(obj_type, cls.BUILDERS)

    def expand(self, node: Any) -> Iterable[Any]:
        expander = self.resolve_expander(type(node))
        if expander is None:
            return self.default_expander(node)
        return expander(self, node)

    def build(self, node: Any, children: List[TreeNode]) -> TreeNode:
        builder = self.resolve_builder(type(node))
        if builder is None:
            result = self.default_builder(node, children)
        else:
            result = builder(self, node, children)
        if not isinstance(result, TreeNode):
            if builder is None:
                source = f"{self.__class__.__name__}.default_builder"
            else:
                source = f"{builder!r}"
            raise ValueError(f"{source}(node={node!r}, children={children!r}) returned {result!r}; "
                             f"builders must return a graphtage.TreeNode")
        return result

    def build_tree(self, root_obj) -> TreeNode:
        children = self.expand(root_obj)
        work: List[Tuple[Any, List[TreeNode], List[Any]]] = [(root_obj, [], list(reversed(list(children))))]
        basic_builder = BasicBuilder(self.options)
        with self.options.printer.tqdm(
                desc="Walking the Tree", leave=False, delay=2.0, unit=" nodes", total=1 + len(work[-1][-1])
        ) as t:
            while work:
                node, processed_children, unprocessed_children = work[-1]

                if unprocessed_children:
                    child = unprocessed_children.pop()
                    t.update(1)

                    grandchildren = list(self.expand(child))

                    if grandchildren and self.options.check_for_cycles:
                        # first, check if all of our grandchildren are leaves; if so, we don't need to check for a cycle
                        all_are_leaves = all(
                            all(False for _ in self.expand(grandchild))
                            for grandchild in grandchildren
                        )
                        if not all_are_leaves:
                            # make sure we aren't already in the process of expanding this child
                            is_cycle = False
                            for already_expanding, _, _ in work:
                                if already_expanding is child:
                                    if self.options.ignore_cycles:
                                        log.debug(f"Detected a cycle in {node!r} at child {child!r}; ignoring…")
                                        processed_children.append(CyclicReference(child))
                                        is_cycle = True
                                        break
                                    else:
                                        raise ValueError(f"Detected a cycle in {node!r} at child {child!r}")
                            if is_cycle:
                                continue
                    work.append((child, [], list(reversed(grandchildren))))
                    t.total = t.total + 1 + len(grandchildren)
                    t.refresh()
                    continue

                _ = work.pop()
                t.update(1)

                new_node = self.build(node, processed_children)
                if not work:
                    return new_node
                work[-1][1].append(new_node)

            return NullNode()


class BasicBuilder(Builder):
    """A builder for basic Python types"""

    @Builder.builder(int)
    def build_int(self, obj: int, _) -> IntegerNode:
        return IntegerNode(obj)

    @Builder.builder(str)
    @Builder.builder(bytes)
    def build_str(self, obj: str, _) -> StringNode:
        return StringNode(obj)

    @Builder.builder(type(None))
    def build_none(self, obj, _) -> NullNode:
        assert obj is None
        return NullNode()

    @Builder.builder(float)
    def build_float(self, obj: float, _) -> FloatNode:
        return FloatNode(obj)

    @Builder.builder(bool)
    def build_bool(self, obj: bool, _) -> BoolNode:
        return BoolNode(obj)

    @Builder.expander(list)
    @Builder.expander(tuple)
    @Builder.expander(set)
    @Builder.expander(frozenset)
    def expand_list(self, obj: list):
        yield from obj

    @Builder.builder(list)
    @Builder.builder(tuple)
    def build_list(self, obj, children: List[TreeNode]) -> ListNode:
        return ListNode(
            children,
            allow_list_edits=self.options.allow_list_edits,
            allow_list_edits_when_same_length=self.options.allow_list_edits_when_same_length
        )

    @Builder.builder(set)
    @Builder.builder(frozenset)
    def build_set(self, obj, children: List[TreeNode]) -> MultiSetNode:
        return MultiSetNode(children)

    @Builder.expander(dict)
    def expand_dict(self, obj: dict):
        yield from obj.keys()
        yield from obj.values()

    @Builder.builder(dict)
    def build_dict(self, _, children: List[TreeNode]):
        n = len(children) // 2
        keys = children[:n]
        values = children[n:]
        dict_items = {
            k: v
            for k, v in zip(keys, values)
        }
        if self.options.allow_key_edits:
            dict_node = DictNode.from_dict(dict_items)
            dict_node.auto_match_keys = self.options.auto_match_keys
            return dict_node
        else:
            return FixedKeyDictNode.from_dict(dict_items)
