from io import StringIO
from unittest import TestCase

from graphtage import IntegerNode, Replace, StringNode
from graphtage.dataclasses import DataClassEdit, DataClassNode
from graphtage.printer import Fore, Printer


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

    def test_print_renders_slots(self):
        """:meth:`DataClassNode.print` is the fallback when no formatter resolves the node type."""
        class Foo(DataClassNode):
            name: StringNode
            count: IntegerNode

        stream = StringIO()
        Foo(name=StringNode("x"), count=IntegerNode(3)).print(Printer(out_stream=stream, ansi_color=False))
        self.assertEqual('Foo(name="x", count=3)', stream.getvalue())

    def test_print_colors_the_class_name_yellow(self):
        """The class name must be yellow; ``Fore.Yellow`` used to raise :exc:`AttributeError` here."""
        class Foo(DataClassNode):
            name: StringNode

        stream = StringIO()
        Foo(name=StringNode("x")).print(Printer(out_stream=stream, ansi_color=True))
        self.assertIn(f"{Fore.YELLOW}Foo", stream.getvalue())

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
