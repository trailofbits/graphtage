"""
A data structure that can hold a set of unique Python objects, even if those objects are not hashable.
Uniqueness is determined based upon equality, if possible, rather than identity.
"""

from collections.abc import MutableSet
from typing import Any, Hashable, Iterable, Set


class _HashableWrapper:
    def __init__(self, obj):
        self.obj = obj

    def __hash__(self):
        return 0

    def __eq__(self, other):
        if not isinstance(other, _HashableWrapper):
            return False
        try:
            return self.obj == other.obj
        except:
            return id(self.obj) == id(other.obj)


class ObjectSet(MutableSet):
    """A set that can hold unhashable Python objects

    Uniqueness is determined based upon equality, if possible, rather than identity.

    """
    def __init__(self, initial_objs: Iterable[Any] = ()):
        self.objs: Set[Any] = set()
        for obj in initial_objs:
            self.add(obj)

    def add(self, value):
        if not isinstance(value, Hashable):
            value = _HashableWrapper(value)
        self.objs.add(value)

    def discard(self, value):
        if not isinstance(value, Hashable):
            value = _HashableWrapper(value)
        self.objs.remove(value)

    def __contains__(self, x):
        if not isinstance(x, Hashable):
            x = _HashableWrapper(x)
        return x in self.objs

    def __len__(self):
        return len(self.objs)

    def __iter__(self):
        for obj in self.objs:
            if isinstance(obj, _HashableWrapper):
                yield obj.obj
            else:
                yield obj
