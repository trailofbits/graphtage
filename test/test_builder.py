from inspect import signature
from unittest import TestCase

from graphtage import BuildOptions, IntegerNode, ListNode, TreeNode, UnorderedListNode
from graphtage.builder import BasicBuilder, Builder


class TestBuilder(TestCase):
    def test_basic_builder(self):
        result = BasicBuilder().build_tree([1, "a", (2, "b"), {1, 2}, {"a": "b"}, None])
        self.assertIsInstance(result, ListNode)
        self.assertEqual(6, len(result.children()))

    def test_ignore_list_order(self):
        options = BuildOptions(ignore_list_order=True)
        result = BasicBuilder(options).build_tree([1, 2, 3])
        self.assertIsInstance(result, UnorderedListNode)
        self.assertEqual(3, len(result.children()))
        self.assertEqual([1, 2, 3], result.to_obj())

    def test_custom_builder(self):
        test = self

        class Foo:
            def __init__(self, bar):
                self.bar = bar

        class Tester(BasicBuilder):
            @Builder.expander(Foo)
            def expand_foo(self, obj: Foo):
                yield obj.bar

            @Builder.builder(Foo)
            def build_foo(self, obj: Foo, children: list[TreeNode]):
                test.assertEqual(1, len(children))
                return children[0]

        tree = Tester().build_tree(Foo(10))
        self.assertIsInstance(tree, IntegerNode)
        self.assertEqual(10, tree.object)


class TestBuildOptions(TestCase):
    def test_check_for_cycles(self):
        parameters = signature(BuildOptions.__init__).parameters
        self.assertIn("check_for_cycles", parameters)
        self.assertNotIn("check_for_cyces", parameters)
        self.assertTrue(BuildOptions().check_for_cycles)
        self.assertFalse(BuildOptions(check_for_cycles=False).check_for_cycles)

    def test_deprecated_check_for_cycles_spelling(self):
        with self.assertWarns(DeprecationWarning):
            options = BuildOptions(check_for_cyces=False)
        self.assertFalse(options.check_for_cycles)
        with self.assertWarns(DeprecationWarning):
            options = BuildOptions(check_for_cyces=True)
        self.assertTrue(options.check_for_cycles)

    def test_check_for_cycles_sets_no_misspelled_attribute(self):
        self.assertNotIn("check_for_cyces", vars(BuildOptions(check_for_cycles=False)))
        with self.assertWarns(DeprecationWarning):
            options = BuildOptions(check_for_cyces=False)
        self.assertNotIn("check_for_cyces", vars(options))

    def test_conflicting_check_for_cycles_spellings(self):
        with self.assertRaises(TypeError):
            BuildOptions(check_for_cycles=True, check_for_cyces=False)
