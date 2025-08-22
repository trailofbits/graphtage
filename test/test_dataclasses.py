from unittest import TestCase

from graphtage import IntegerNode, Replace, StringNode
from graphtage.dataclasses import DataClassEdit, DataClassNode


class TestDataclasses(TestCase):
    def test_inheritance(self):
        class Foo(DataClassNode):
            foo: IntegerNode
            initialized = False

            def post_init(self):
                self.initialized = True

        class Bar(Foo):
            bar: StringNode
            initialized = False

            def post_init(self):
                self.initialized = True

        self.assertEqual(("foo",), Foo._SLOTS)
        self.assertEqual(0, len(Foo._DATA_CLASS_ANCESTORS))
        self.assertEqual(("foo", "bar",), Bar._SLOTS)
        self.assertEqual(1, len(Bar._DATA_CLASS_ANCESTORS))

        b = Bar(foo=IntegerNode(10), bar=StringNode("bar"))
        self.assertEqual(10, b.foo.object)
        self.assertEqual("bar", b.bar.object)
        self.assertTrue(b.initialized)

        # now test a mixture of positional and keyword arguments
        b = Bar(StringNode("bar"), foo=IntegerNode(10))
        self.assertEqual(10, b.foo.object)
        self.assertEqual("bar", b.bar.object)
        self.assertTrue(b.initialized)

        # test equality
        self.assertEqual(Bar(IntegerNode(10), StringNode("bar")), b)
        self.assertNotEqual(Bar(IntegerNode(11), StringNode("bar")), b)

        # test diffing of different dataclasses
        f = Foo(IntegerNode(10))
        edit = f.edits(b)
        self.assertIsInstance(edit, Replace)
        c = Foo(IntegerNode(12))
        edit = f.edits(c)
        self.assertIsInstance(edit, DataClassEdit)

    def test_inheritance_with_duplicate(self):
        def define_duplicate():
            class BaseFoo(DataClassNode):
                foo: StringNode

            class DuplicateFoo(BaseFoo):
                bar: IntegerNode
                foo: IntegerNode

        self.assertRaises(TypeError, define_duplicate)

    def test_runtime_type_checking(self):
        class Foo(DataClassNode):
            foo: IntegerNode

        def try_wrong_type():
            return Foo(StringNode("foo"))

        self.assertRaises(ValueError, try_wrong_type)
