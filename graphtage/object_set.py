"""
A data structure that can hold a set of unique Python objects, even if those objects are not hashable.
Uniqueness is determined based upon identity.
"""

from collections.abc import MutableSet
from typing import Any, Iterable, Set


class IdentityHash:
    def __init__(self, obj):
        self.obj = obj

    def __hash__(self):
        return id(self.obj)

    def __eq__(self, other):
        if not isinstance(other, IdentityHash):
            return False
        return id(self.obj) == id(other.obj)


class ObjectSet(MutableSet):
    """A set that can hold unhashable Python objects

    Uniqueness is determined based upon identity.

    """
    def __init__(self, initial_objs: Iterable[Any] = ()):
        self.objs: Set[IdentityHash] = set()
        for obj in initial_objs:
            self.add(obj)

    def add(self, value):
        self.objs.add(IdentityHash(value))

    def discard(self, value):
        value = IdentityHash(value)
        self.objs.remove(value)

    def __contains__(self, x):
        x = IdentityHash(x)
        return x in self.objs

    def __len__(self):
        return len(self.objs)

    def __iter__(self):
        for obj in self.objs:
            yield obj.obj

    def __str__(self):
        return f"{{{', '.join(map(str, self.objs))}}}"
