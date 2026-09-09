from collections.abc import Callable, Iterator
from typing import get_origin

from . import AbstractCompoundEdit, Edit, Range, Replace
from .printer import Fore, Printer
from .tree import ContainerNode, GraphtageFormatter, TreeNode


class DataClassEdit(AbstractCompoundEdit):
    def __init__(self, from_node: "DataClassNode", to_node: "DataClassNode"):
        from_slots = dict(from_node.items())
        to_slots = dict(to_node.items())
        if from_slots.keys() != to_slots.keys():
            raise ValueError(f"Node {from_node!r} cannot be edited to {to_node!r} because they have incompatible slots")
        self.slot_edits: list[Edit] = [
            value.edits(to_slots[slot])
            for slot, value in from_slots.items()
        ]
        super().__init__(from_node, to_node)

    def bounds(self) -> Range:
        total = Range(0, 0)
        for e in self.slot_edits:
            total = total + e.bounds()
        return total

    def edits(self) -> Iterator[Edit]:
        yield from self.slot_edits

    def tighten_bounds(self) -> bool:
        return any(edit.tighten_bounds() for edit in self.slot_edits)

    def print(self, formatter: GraphtageFormatter, printer: Printer):
        """Prints this edit by delegating to the formatter for the node being edited.

        The default :meth:`graphtage.AbstractCompoundEdit.print` implementation prints the slot edits back to back,
        which drops whatever syntax the node's formatter writes between the slots. Delegating to the node formatter
        keeps that syntax, and the formatter reaches the slot edits as it prints each child.

        This is equivalent to::

            formatter.get_formatter(self.from_node)(printer, self.from_node)

        """
        formatter.get_formatter(self.from_node)(printer, self.from_node)


class DataClassNode(ContainerNode):
    """A container node that can be initialized similar to a Python :func:`dataclasses.dataclass`"""

    _SLOTS: tuple[str, ...]
    _SLOT_ANNOTATIONS: dict[str, type[TreeNode]]
    _DATA_CLASS_ANCESTORS: list[type["DataClassNode"]]
    _POST_INITS: tuple[Callable[["DataClassNode"], None], ...]

    def __init__(self, *args, **kwargs):
        """Be careful extending __init__; consider using :func:`DataClassNode.post_init` instead."""
        our_kwargs = {
            k: v
            for k, v in kwargs.items()
            if k in self._SLOTS
        }
        parent_kwargs = {
            k: v
            for k, v in kwargs.items()
            if k not in self._SLOTS
        }
        required_positional_args = len(self._SLOTS) - len(our_kwargs)
        assert required_positional_args >= 0
        if required_positional_args > len(args):
            raise ValueError(f"Not enough arguments sent to {self.__class__.__name__}.__init__: {args!r} {kwargs!r}; "
                             f"expected at least {len(self._SLOTS)}")
        start_index = len(args) - required_positional_args
        parent_args = args[:start_index]
        super().__init__(*parent_args, **parent_kwargs)
        our_args = list(args[start_index:])
        for s in self._SLOTS:
            if s in our_kwargs:
                value = our_kwargs[s]
            elif not our_args:
                raise ValueError(f"Missing argument for {self.__class__.__name__}.{s}")
            else:
                value = our_args[0]
                our_args = our_args[1:]
            expected_type = self._SLOT_ANNOTATIONS[s]
            if not isinstance(value, expected_type):
                raise ValueError(f"Expected a node of type {expected_type.__name__} for argument "
                                 f"{self.__class__.__name__}.{s} but instead got {value!r}")
            setattr(self, s, value)
        # self.__hash__ gets called so often, we cache the result:
        self.__hash = hash(tuple(self))
        for post_init in self._POST_INITS:
            post_init(self)

    def post_init(self):
        """Callback called after this node's slots have been initialized.

        This callback should not call `super().post_init()`. Every implementation in the class hierarchy is called
        automatically, starting with the least derived data class and ending with the class being instantiated. An
        implementation that a subclass inherits without overriding is called only once.
        """
        pass

    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)
        ancestors = [
            c
            for c in reversed(cls.__mro__)
            if c is not cls and issubclass(c, DataClassNode) and c is not DataClassNode
        ]
        cls._DATA_CLASS_ANCESTORS = ancestors
        # Selecting on __dict__ keeps an inherited implementation from being called once per class that inherits it.
        cls._POST_INITS = tuple(
            c.__dict__["post_init"]
            for c in (*ancestors, cls)
            if "post_init" in c.__dict__
        )
        ancestor_slot_names = {
            name: a
            for a in ancestors
            for name in a._SLOTS
        }
        if not hasattr(cls, "_SLOT_ANNOTATIONS") or cls._SLOT_ANNOTATIONS is None:
            cls._SLOT_ANNOTATIONS = {}
            cls._SLOTS = ()
        else:
            cls._SLOT_ANNOTATIONS = dict(cls._SLOT_ANNOTATIONS)
        new_slots = []
        for name, slot_type in cls.__annotations__.items():
            # get_origin() screens out subscripted generics before issubclass() sees them. On Python
            # 3.10 isinstance(list[int], type) is True, so issubclass() would raise there.
            if get_origin(slot_type) is not None:
                continue
            if not isinstance(slot_type, type) or not issubclass(slot_type, TreeNode):
                continue
            if name in ancestor_slot_names:
                raise TypeError(f"Dataclass {cls.__name__} cannot redefine slot {name!r} because it is already "
                                f"defined in its superclass {ancestor_slot_names[name].__name__}")
            new_slots.append(name)
            cls._SLOT_ANNOTATIONS[name] = slot_type
        cls._SLOTS = cls._SLOTS + tuple(new_slots)

    def __hash__(self):
        return self.__hash

    def __iter__(self) -> Iterator[TreeNode]:
        for _, value in self.items():
            yield value

    def items(self) -> Iterator[tuple[str, TreeNode]]:
        for slot in self._SLOTS:
            yield slot, getattr(self, slot)

    def to_obj(self):
        return {
            slot: getattr(self, slot).to_obj()
            for slot in self._SLOTS
        }

    def edits(self, node: TreeNode) -> Edit:
        if isinstance(node, DataClassNode):
            our_slots = set(self._SLOTS)
            their_slots = set(node._SLOTS)
            if our_slots == their_slots:
                return DataClassEdit(self, node)
        return Replace(self, node)

    def calculate_total_size(self) -> int:
        return sum(s.calculate_total_size() for s in self)

    def print(self, printer: Printer):
        with printer.color(Fore.YELLOW):
            printer.write(self.__class__.__name__)
        printer.write("(")
        for i, slot in enumerate(self._SLOTS):
            if i > 0:
                printer.write(", ")
            with printer.color(Fore.RED):
                printer.write(slot)
            with printer.bright():
                printer.write("=")
            getattr(self, slot).print(printer)
        printer.write(")")

    def __len__(self):
        return len(self._SLOTS)

    def __eq__(self, other):
        return isinstance(other, DataClassNode) and dict(self.items()) == dict(other.items())

    def __repr__(self):
        attrs = ", ".join(
            f"{slot}={value!r}"
            for slot, value in self.items()
        )
        return f"{self.__class__.__name__}({attrs})"
