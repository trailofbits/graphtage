from typing import Dict, Iterator, List, Tuple, Type

from . import AbstractCompoundEdit, Edit, Range, Replace
from .printer import Fore, Printer
from .tree import ContainerNode, TreeNode


class DataClassEdit(AbstractCompoundEdit):
    def __init__(self, from_node: "DataClassNode", to_node: "DataClassNode"):
        from_slots = dict(from_node.items()).keys()
        if not from_slots == dict(to_node.items()).keys():
            raise ValueError(f"Node {from_node!r} cannot be edited to {to_node!r} because they have incompatible slots")
        self.slot_edits: List[Edit] = [
            getattr(from_node, slot).edits(getattr(to_node, slot))
            for slot in from_slots
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
        for edit in self.slot_edits:
            if edit.tighten_bounds():
                return True
        return False


class DataClassNode(ContainerNode):
    """A container node that can be initialized similar to a Python :func:`dataclasses.dataclass`"""

    _SLOTS: Tuple[str, ...]
    _SLOT_ANNOTATIONS: Dict[str, Type[TreeNode]]
    _NUM_ANCESTOR_SLOTS: int

    def __init__(self, *args, **kwargs):
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
        starting_index = self._NUM_ANCESTOR_SLOTS - len(parent_kwargs)
        if starting_index < 0:
            raise ValueError(f"Unexpected number of kwargs sent to {self.__class__.__name__}.__init__: {kwargs!r}")
        parent_args = args[:starting_index]
        super().__init__(*parent_args, **parent_kwargs)
        our_args = list(args[starting_index:])
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

    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)
        ancestors = [
            c
            for c in cls.__mro__
            if c is not cls and issubclass(c, DataClassNode) and c is not DataClassNode
        ]
        ancestor_slot_names = {
            name: a
            for a in ancestors
            for name in a._SLOTS
        }
        setattr(cls, "_NUM_ANCESTOR_SLOTS", sum(
            len(c._SLOTS)
            for c in ancestors
        ))
        if not hasattr(cls, "_SLOTS") or cls._SLOTS is None:
            cls._SLOTS = ()
            cls._SLOT_ANNOTATIONS = {}
        else:
            cls._SLOT_ANNOTATIONS = dict(cls._SLOT_ANNOTATIONS)
        new_slots = []
        for i, (name, slot_type) in enumerate(cls.__annotations__.items()):
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
        try:
            yield from super().__iter__()
        except NotImplementedError:
            pass
        for slot in self._SLOTS:
            yield getattr(self, slot)

    def items(self) -> Iterator[Tuple[str, TreeNode]]:
        if hasattr(super(), "items"):
            yield from super().items()  # type: ignore
        for slot in self._SLOTS:
            yield slot, getattr(self, slot)

    def to_obj(self):
        return {
            slot: getattr(self, slot).to_obj()
            for slot in self._SLOTS
        }

    def edits(self, node: TreeNode) -> Edit:
        if isinstance(node, DataClassNode):
            our_slots = dict(self.items()).keys()
            their_slots = dict(node.items()).keys()
            if our_slots == their_slots:
                return DataClassEdit(self, node)
        return Replace(self, node)

    def calculate_total_size(self) -> int:
        return sum(s.calculate_total_size() for s in self)

    def print(self, printer: Printer):
        with printer.color(Fore.Yellow):
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
        return self._NUM_ANCESTOR_SLOTS + len(self._SLOTS)

    def __eq__(self, other):
        return isinstance(other, DataClassNode) and dict(self.items()) == dict(other.items())

    def __repr__(self):
        attrs = ", ".join(
            f"{slot}={value}"
            for slot, value in self.items()
        )
        return f"{self.__class__.__name__}({attrs})"
