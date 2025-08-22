from unittest import TestCase

from graphtage.object_set import ObjectSet


class UnhashableWithBrokenEquality:
    def __init__(self, value):
        self.value = value

    def __eq__(self, other):
        raise ValueError()


class Unhashable(UnhashableWithBrokenEquality):
    def __eq__(self, other):
        return isinstance(other, Unhashable) and self.value == other.value


class TestObjectSet(TestCase):
    def test_unhashability(self):
        self.assertRaises(TypeError, lambda: hash(Unhashable(10)))

    def test_object_set(self):
        u = Unhashable(10)
        u2 = Unhashable(11)
        objs = ObjectSet((10, u, u2))
        self.assertIn(10, objs)
        self.assertIn(u, objs)
        self.assertIn(u2, objs)
        self.assertEqual(3, len(objs))
        objs.remove(u)
        self.assertIn(10, objs)
        self.assertNotIn(u, objs)
        self.assertIn(u2, objs)
        self.assertEqual(2, len(objs))

    def test_broken_equality(self):
        u = UnhashableWithBrokenEquality(10)
        u2 = UnhashableWithBrokenEquality(10)
        # this will default to uniqueness by identity
        objs = ObjectSet((10, u, u2))
        self.assertIn(10, objs)
        self.assertIn(u, objs)
        self.assertIn(u2, objs)
        self.assertEqual(3, len(objs))
