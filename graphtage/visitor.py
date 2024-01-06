"""A module intended to simplify building Graphtage IR trees from other tree-like data structures."""

from abc import ABC
import logging
from typing import Any, Callable, Dict, Iterable, List, Optional, Tuple, Type, TypeVar

from . import BoolNode, BuildOptions, FloatNode, IntegerNode, LeafNode, ListNode, NullNode, StringNode, TreeNode
from .object_set import ObjectSet

C = TypeVar("C")
T = TypeVar("T")

log = logging.getLogger(__name__)


class CyclicReference(LeafNode):
    def __hash__(self):
        return id(self.object)

    def __eq__(self, other):
        return isinstance(other, CyclicReference) and other.object is self.object


class Visitor(ABC):
    EXPANDERS: Dict[Type[Any], Callable[["Visitor", Any], Optional[Iterable[Any]]]]
    BUILDERS: Dict[Type[Any], Callable[["Visitor", Any, List[TreeNode]], TreeNode]]

    def __init__(self, options: Optional[BuildOptions] = None):
        if options is None:
            self.options: BuildOptions = BuildOptions()
        else:
            self.options = options

    @staticmethod
    def expander(node_type: Type[T]):
        def wrapper(func: Callable[[C, T], Iterable[Any]]) -> Callable[[C, T], Iterable[Any]]:
            setattr(func, "_visitor_expander_for_type", node_type)
            return func

        return wrapper

    @staticmethod
    def builder(node_type: Type[T]):
        def wrapper(func: Callable[[C, T, List[TreeNode]], TreeNode]) -> Callable[[C, T, List[TreeNode]], TreeNode]:
            setattr(func, "_visitor_builder_for_type", node_type)
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
        for member_name in dir(cls):
            try:
                member = getattr(cls, member_name)
            except AttributeError:
                continue
            if hasattr(member, "_visitor_expander_for_type"):
                expander_type = getattr(member, "_visitor_expander_for_type")
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
                builder_type = getattr(member, "_visitor_builder_for_type")
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

    def default_expander(self, node: Any) -> Optional[Iterable[Any]]:
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
            return self.default_builder(node, children)
        return builder(self, node, children)

    def build_tree(self, root_obj) -> TreeNode:
        children = self.expand(root_obj)
        work: List[Tuple[Any, List[TreeNode], List[Any]]] = [(root_obj, [], list(reversed(list(children))))]
        history = ObjectSet((root_obj,))
        while work:
            node, processed_children, unprocessed_children = work[-1]

            if unprocessed_children:
                child = unprocessed_children.pop()

                grandchildren = list(self.expand(child))

                if grandchildren and self.options.check_for_cycles:
                    if child in history:
                        if self.options.ignore_cycles:
                            log.debug(f"Detected a cycle in {node!r} at child {child!r}; ignoring…")
                            processed_children.append(CyclicReference(child))
                            continue
                        else:
                            raise ValueError(f"Detected a cycle in {node!r} at child {child!r}")
                    history.add(child)

                work.append((child, [], list(reversed(grandchildren))))
                continue

            _ = work.pop()

            new_node = self.build(node, processed_children)
            if not work:
                return new_node
            work[-1][1].append(new_node)

        return NullNode()


class AbstractVisitor(Visitor):
    @Visitor.builder(int)
    def build_int(self, obj: int, _) -> IntegerNode:
        return IntegerNode(obj)

    @Visitor.builder(str)
    def build_str(self, obj: str, _) -> StringNode:
        return StringNode(obj)

    @Visitor.builder(type(None))
    def build_none(self, obj, _) -> NullNode:
        assert obj is None
        return NullNode()

    @Visitor.builder(float)
    def build_float(self, obj: float, _) -> FloatNode:
        return FloatNode(obj)

    @Visitor.builder(bool)
    def build_bool(self, obj: bool, _) -> BoolNode:
        return BoolNode(obj)

    @Visitor.expander(list)
    def list_expander(self, obj: list):
        yield from obj

    @Visitor.builder(list)
    def list_builder(self, obj: list, children: List[TreeNode]) -> TreeNode:
        return ListNode(children)
