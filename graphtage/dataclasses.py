from typing import Dict, Iterator, List, Tuple, Type

from . import AbstractCompoundEdit, Edit, Range, Replace
from .printer import Fore, Printer
from .tree import ContainerNode, TreeNode


class DataClassEdit(AbstractCompoundEdit):
    def __init__(self, from_node: "DataClassNode", to_node: "DataClassNode"):
        from_slots = dict(from_node.items())
        to_slots = dict(to_node.items())
        if from_slots.keys() != to_slots.keys():
            raise ValueError(f"Node {from_node!r} cannot be edited to {to_node!r} because they have incompatible slots")
        self.slot_edits: List[Edit] = [
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
        for edit in self.slot_edits:
            if edit.tighten_bounds():
                return True
        return False


class DataClassNode(ContainerNode):
    """A container node that can be initialized similar to a Python :func:`dataclasses.dataclass`"""

    _SLOTS: Tuple[str, ...]
    _SLOT_ANNOTATIONS: Dict[str, Type[TreeNode]]
    _DATA_CLASS_ANCESTORS: List[Type["DataClassNode"]]

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
        for ancestor in self._DATA_CLASS_ANCESTORS:
            ancestor.post_init(self)

    def post_init(self):
        """Callback called after this class's members have been initialized.

        This callback should not call `super().post_init()`. Each superclass's `post_init()` will be automatically
        called in order of the `__mro__`.
        """
        pass

    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)
        ancestors = [
            c
            for c in cls.__mro__
            if c is not cls and issubclass(c, DataClassNode) and c is not DataClassNode
        ]
        cls._DATA_CLASS_ANCESTORS = ancestors
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
        for _, value in self.items():
            yield value

    def items(self) -> Iterator[Tuple[str, TreeNode]]:
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
        return len(self._SLOTS)

    def __eq__(self, other):
        return isinstance(other, DataClassNode) and dict(self.items()) == dict(other.items())

    def __repr__(self):
        attrs = ", ".join(
            f"{slot}={value!r}"
            for slot, value in self.items()
        )
        return f"{self.__class__.__name__}({attrs})"
