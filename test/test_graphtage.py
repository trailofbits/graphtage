import random
from io import StringIO
from unittest import TestCase

import graphtage
import graphtage.json
import graphtage.multiset
import graphtage.tree
from graphtage.printer import Printer

from .timing import run_with_time_limit


def unordered(*values) -> graphtage.UnorderedListNode:
    return graphtage.UnorderedListNode([graphtage.IntegerNode(v) for v in values])


def ordered(*values) -> graphtage.ListNode:
    return graphtage.ListNode([graphtage.IntegerNode(v) for v in values])


class TestGraphtage(TestCase):
    @classmethod
    def setUpClass(cls):
        cls.small_from = graphtage.json.build_tree({
            "test": "foo",
            "baz": 1
        })
        cls.small_to = graphtage.json.build_tree({
            "test": "bar",
            "baz": 2
        })
        cls.list_from = graphtage.json.build_tree([0, 1, 2, 3, 4, 5])
        cls.list_to = graphtage.json.build_tree([1, 2, 3, 4, 5])

    def test_string_diff_printing(self):
        s1 = graphtage.StringNode("abcdef")
        s2 = graphtage.StringNode("azced")
        diff = s1.diff(s2)
        out_stream = StringIO()
        p = Printer(ansi_color=True, out_stream=out_stream)
        diff.print(p)
        self.assertEqual(diff.edited_cost(), 3)
        self.assertEqual('\x1b[32m"\x1b[37m\x1b[41m\x1b[1m\x1b[0m\x1b[49m\x1b[32m\x1b[37m\x1b[42m\x1b[1m\x1b[0m\x1b[49m\x1b[32ma\x1b[37m\x1b[41m\x1b[1mb̶\x1b[0m\x1b[49m\x1b[32m\x1b[37m\x1b[42m\x1b[1mz̟\x1b[0m\x1b[49m\x1b[32mc\x1b[37m\x1b[41m\x1b[1m\x1b[0m\x1b[49m\x1b[32m\x1b[37m\x1b[42m\x1b[1m\x1b[0m\x1b[49m\x1b[32m\x1b[37m\x1b[41m\x1b[1md̶\x1b[0m\x1b[49m\x1b[32m\x1b[37m\x1b[42m\x1b[1m\x1b[0m\x1b[49m\x1b[32me\x1b[37m\x1b[41m\x1b[1mf̶\x1b[0m\x1b[49m\x1b[32m\x1b[37m\x1b[42m\x1b[1md̟\x1b[0m\x1b[49m\x1b[32m"\x1b[39m', out_stream.getvalue())

    def test_string_diff_substitution_run(self):
        s1 = graphtage.StringNode('abcdefg')
        s2 = graphtage.StringNode('abhijfg')
        diff = s1.diff(s2)
        out_stream = StringIO()
        p = Printer(ansi_color=True, out_stream=out_stream)
        diff.print(p)
        self.assertEqual(diff.edited_cost(), 3)
        self.assertEqual('\x1b[32m"\x1b[37m\x1b[41m\x1b[1m\x1b[0m\x1b[49m\x1b[32m\x1b[37m\x1b[42m\x1b[1m\x1b[0m\x1b[49m\x1b[32ma\x1b[37m\x1b[41m\x1b[1m\x1b[0m\x1b[49m\x1b[32m\x1b[37m\x1b[42m\x1b[1m\x1b[0m\x1b[49m\x1b[32mb\x1b[37m\x1b[41m\x1b[1mc̶d̶e̶\x1b[0m\x1b[49m\x1b[32m\x1b[37m\x1b[42m\x1b[1mh̟i̟j̟\x1b[0m\x1b[49m\x1b[32mf\x1b[37m\x1b[41m\x1b[1m\x1b[0m\x1b[49m\x1b[32m\x1b[37m\x1b[42m\x1b[1m\x1b[0m\x1b[49m\x1b[32mg\x1b[37m\x1b[41m\x1b[1m\x1b[0m\x1b[49m\x1b[32m\x1b[37m\x1b[42m\x1b[1m\x1b[0m\x1b[49m\x1b[32m"\x1b[39m', out_stream.getvalue())

    def test_small_diff(self):
        diff = self.small_from.diff(self.small_to)
        self.assertIsInstance(diff, graphtage.DictNode)
        self.assertIsInstance(diff, graphtage.tree.EditedTreeNode)
        self.assertEqual(1, len(diff.edit_list))
        self.assertIsInstance(diff.edit_list[0], graphtage.multiset.MultiSetEdit)
        has_test_match = False
        has_baz_match = False
        for edit in diff.edit_list[0].edits():
            if edit.bounds().upper_bound > 0:
                self.assertIsInstance(edit, graphtage.KeyValuePairEdit)
                key_edit = edit.key_edit
                value_edit = edit.value_edit
                if isinstance(value_edit.from_node, graphtage.StringNode):
                    self.assertIsInstance(key_edit.to_node, graphtage.StringNode)
                    self.assertEqual(key_edit.from_node.object, 'test')
                    self.assertEqual(value_edit.from_node.object, 'foo')
                    self.assertEqual(value_edit.to_node.object, 'bar')
                    self.assertEqual(edit.bounds().upper_bound, 3)
                    self.assertFalse(has_test_match)
                    has_test_match = True
                elif isinstance(value_edit.from_node, graphtage.IntegerNode):
                    self.assertIsInstance(value_edit.to_node, graphtage.IntegerNode)
                    self.assertEqual(value_edit.from_node.object, 1)
                    self.assertEqual(value_edit.to_node.object, 2)
                    self.assertEqual(value_edit.bounds().upper_bound, 1)
                    self.assertFalse(has_baz_match)
                    has_baz_match = True
                else:
                    self.fail()
        self.assertTrue(has_test_match)
        self.assertTrue(has_baz_match)

    def test_list_diff(self):
        diff = self.list_from.diff(self.list_to)
        self.assertIsInstance(diff, graphtage.ListNode)
        self.assertIsInstance(diff, graphtage.tree.EditedTreeNode)
        self.assertEqual(1, len(diff.edit_list))
        self.assertIsInstance(diff.edit_list[0], graphtage.EditDistance)
        self.assertEqual(1, diff.edited_cost())
        num_removals = 0
        for edit in diff.edit_list[0].edits():
            if edit.bounds().upper_bound > 0:
                self.assertIsInstance(edit, graphtage.Remove)
                self.assertIsInstance(edit.from_node, graphtage.IntegerNode)
                self.assertEqual(edit.from_node.object, 0)
                self.assertIsInstance(edit.to_node, graphtage.ListNode)
                self.assertEqual(edit.to_node, self.list_from)
                num_removals += 1
            else:
                self.assertIsInstance(edit, graphtage.Match)
        self.assertEqual(1, num_removals)

    def test_list_diff_optimality(self):
        """Reproduces https://github.com/trailofbits/graphtage/issues/89"""
        diff = graphtage.json.build_tree(["de", 1]).diff(graphtage.json.build_tree([2]))
        self.assertEqual(3, diff.edited_cost())

    def test_single_element_list(self):
        diff = graphtage.json.build_tree([1]).diff(graphtage.json.build_tree([2]))
        self.assertIsInstance(diff, graphtage.ListNode)
        self.assertIsInstance(diff, graphtage.tree.EditedTreeNode)
        self.assertEqual(1, len(diff.edit_list))
        self.assertIsInstance(diff.edit_list[0], graphtage.FixedLengthSequenceEdit)

    def test_empty_list(self):
        diff = graphtage.ListNode(()).diff(graphtage.ListNode(()))
        self.assertEqual(1, len(diff.edit_list))
        self.assertIsInstance(diff.edit_list[0], graphtage.Match)
        self.assertEqual(0, diff.edit_list[0].bounds().upper_bound)

    def test_null_json(self):
        diff = graphtage.json.build_tree([None]).diff(graphtage.json.build_tree([1]))
        self.assertIsInstance(diff, graphtage.ListNode)
        self.assertIsInstance(diff, graphtage.tree.EditedTreeNode)
        self.assertEqual(1, len(diff.edit_list))
        self.assertIsInstance(diff.edit_list[0], graphtage.FixedLengthSequenceEdit)


class TestUnorderedListNode(TestCase):
    def test_ordered_list_reorder_costs(self):
        """Reordering an ordinary list is an edit; this is what :class:`graphtage.UnorderedListNode` changes."""
        edit = ordered(1, 2, 3).edits(ordered(3, 2, 1))
        self.assertIsInstance(edit, graphtage.EditDistance)
        self.assertEqual(2, ordered(1, 2, 3).diff(ordered(3, 2, 1)).edited_cost())

    def test_reorder_is_free(self):
        edit = unordered(1, 2, 3).edits(unordered(3, 2, 1))
        self.assertIsInstance(edit, graphtage.Match)
        self.assertEqual(0, edit.bounds().upper_bound)
        self.assertEqual(0, unordered(1, 2, 3).diff(unordered(3, 2, 1)).edited_cost())

    def test_reorder_against_an_ordered_list(self):
        """Only one side has to ignore order, which is what makes cross-format comparisons work."""
        self.assertIsInstance(unordered(1, 2, 3).edits(ordered(3, 2, 1)), graphtage.Match)
        self.assertEqual(0, unordered(1, 2, 3).edits(ordered(3, 2, 1)).bounds().upper_bound)

    def test_reorder_with_a_change(self):
        edit = unordered(1, 2, 3).edits(unordered(3, 9, 1))
        self.assertIsInstance(edit, graphtage.multiset.MultiSetEdit)
        self.assertEqual(1, edit.bounds().upper_bound)

    def test_duplicates_are_counted(self):
        self.assertIsInstance(unordered(1, 1, 2).edits(unordered(2, 1, 1)), graphtage.Match)
        self.assertEqual(0, unordered(1, 1, 2).edits(unordered(2, 1, 1)).bounds().upper_bound)
        self.assertGreater(unordered(1, 1, 2).diff(unordered(1, 2, 2)).edited_cost(), 0)

    def test_nested_lists(self):
        from_node = graphtage.UnorderedListNode([unordered(1, 2), unordered(3, 4)])
        to_node = graphtage.UnorderedListNode([unordered(4, 3), unordered(2, 1)])
        edit = from_node.edits(to_node)
        self.assertIsInstance(edit, graphtage.Match)
        self.assertEqual(0, edit.bounds().upper_bound)

    def test_versus_dict(self):
        """A list is never matched against a dictionary, whether or not its order is ignored."""
        dict_node = graphtage.DictNode.from_dict({graphtage.StringNode("a"): graphtage.IntegerNode(1)})
        self.assertIsInstance(unordered(1, 2, 3).edits(dict_node), graphtage.Replace)
        self.assertIsInstance(dict_node.edits(unordered(1, 2, 3)), graphtage.Replace)

    def test_to_obj_of_unhashable_children(self):
        """:meth:`graphtage.MultiSetNode.to_obj` counts its children, which fails for a list of dictionaries."""
        def children():
            return [
                graphtage.DictNode.from_dict({graphtage.StringNode("a"): graphtage.IntegerNode(1)}),
                graphtage.DictNode.from_dict({graphtage.StringNode("b"): graphtage.IntegerNode(2)}),
            ]

        self.assertEqual([{"a": 1}, {"b": 2}], graphtage.UnorderedListNode(children()).to_obj())
        with self.assertRaises(TypeError):
            graphtage.MultiSetNode(children()).to_obj()

    def test_large_shuffle_is_fast(self):
        """Equal multisets short-circuit to a match, so a big shuffle must not reach the bipartite matcher."""
        values = list(range(1000))
        shuffled = list(values)
        random.Random(56).shuffle(shuffled)
        with run_with_time_limit(seconds=30):
            edit = unordered(*values).edits(unordered(*shuffled))
        self.assertIsInstance(edit, graphtage.Match)
        self.assertEqual(0, edit.bounds().upper_bound)


class TestBytesStringNode(TestCase):
    """Covers :class:`graphtage.StringNode` objects that wrap :class:`bytes` instead of :class:`str`.

    Iterating over :class:`bytes` yields :class:`int` byte values, so the per-character lattice that
    :func:`graphtage.string_edit_distance` builds wraps each byte in a ``StringNode`` holding an ``int``.
    Every test here used to raise ``TypeError: object of type 'int' has no len()``.

    """

    def assert_exact_cost(self, expected: int, from_bytes: str | bytes, to_bytes: str | bytes):
        edit = graphtage.StringNode(from_bytes).edits(graphtage.StringNode(to_bytes))
        while edit.tighten_bounds():
            pass
        self.assertEqual(graphtage.Range(expected, expected), edit.bounds())

    def test_substitution(self):
        """One differing byte costs one, the same as one differing character does."""
        self.assert_exact_cost(1, b"hello", b"hellp")
        self.assert_exact_cost(1, "hello", "hellp")

    def test_equal_bytes_match_for_free(self):
        edit = graphtage.StringNode(b"hello").edits(graphtage.StringNode(b"hello"))
        self.assertIsInstance(edit, graphtage.Match)
        self.assert_exact_cost(0, b"hello", b"hello")

    def test_single_byte(self):
        """A one-byte value takes the single-character shortcut, which is where the crash originated."""
        self.assert_exact_cost(1, b"a", b"b")
        self.assert_exact_cost(0, b"a", b"a")

    def test_empty_bytes(self):
        """An empty value on either side costs one per byte on the other side."""
        self.assert_exact_cost(0, b"", b"")
        self.assert_exact_cost(3, b"", b"abc")
        self.assert_exact_cost(3, b"abc", b"")

    def test_insertion_and_removal_cost_one_per_byte(self):
        """A byte node's total size is one, so inserting or removing it costs the same as a character."""
        self.assertEqual(1, graphtage.StringNode(b"h").total_size)
        self.assertEqual(5, graphtage.StringNode(b"hello").total_size)
        self.assert_exact_cost(1, b"hello", b"hell")
        self.assert_exact_cost(1, b"hell", b"hello")

    def test_str_versus_bytes(self):
        """A ``str`` never equals ``bytes``, so every position counts as a substitution."""
        self.assert_exact_cost(5, "hello", b"hello")
        self.assert_exact_cost(5, b"hello", "hello")
        self.assert_exact_cost(1, "a", b"a")

    @staticmethod
    def render(from_bytes: str | bytes, to_bytes: str | bytes) -> str:
        out_stream = StringIO()
        graphtage.StringNode(from_bytes).diff(graphtage.StringNode(to_bytes)).print(
            Printer(ansi_color=False, out_stream=out_stream)
        )
        return out_stream.getvalue()

    def test_rendering_closes_the_quote(self):
        """The ``b`` prefix used to be repeated before the closing quote, rendering ``b"hellob"``."""
        self.assertEqual('b"hell~~o~~++p++"', self.render(b"hello", b"hellp"))
        self.assertEqual('"hell~~o~~++p++"', self.render("hello", "hellp"))

    def test_rendering_escapes_unprintable_bytes(self):
        self.assertEqual('b"he~~\\x00~~++\\xff++o"', self.render(b"he\x00o", b"he\xffo"))


class StringEditDistanceCounter:
    """Counts how many character-level lattices :class:`graphtage.StringEdit` builds inside a block."""

    def __init__(self):
        self.count = 0
        self._original = graphtage.graphtage.string_edit_distance

    def __enter__(self) -> 'StringEditDistanceCounter':
        def counted(s1, s2):
            self.count += 1
            return self._original(s1, s2)

        graphtage.graphtage.string_edit_distance = counted
        return self

    def __exit__(self, *args):
        graphtage.graphtage.string_edit_distance = self._original


class TestStringEdit(TestCase):
    """Covers the two halves of :class:`graphtage.StringEdit`: its cost and its edit script.

    The cost is computed arithmetically at construction; the lattice that produces the script is built only when
    something reads :attr:`graphtage.StringEdit.edit_distance`. These tests pin that split, because a
    :class:`graphtage.StringEdit` that eagerly builds its lattice is correct but quadratically slower, and a
    lattice whose cost disagrees with the reported cost renders a diff that does not add up to its own price.

    """

    @staticmethod
    def edit(from_str: str | bytes, to_str: str | bytes) -> graphtage.StringEdit:
        return graphtage.StringEdit(graphtage.StringNode(from_str), graphtage.StringNode(to_str))

    def test_bounds_are_definitive_on_construction(self):
        """A caller that never tightens must still see the exact cost.

        :class:`graphtage.matching.WeightedBipartiteMatcher` prices its edges with ``bounds().upper_bound``
        after only partial tightening, so an edit whose initial upper bound is an over-estimate feeds the
        assignment problem a weight that is too large.

        """
        edit = self.edit("kitten", "sitting")
        self.assertTrue(edit.bounds().definitive())
        self.assertEqual(graphtage.Range(3, 3), edit.bounds())
        self.assertEqual(graphtage.Range(3, 3), edit.initial_bounds)

    def test_tighten_bounds_returns_false(self):
        """There is nothing left to tighten, so the loops that drive edits to completion terminate at once."""
        edit = self.edit("kitten", "sitting")
        self.assertFalse(edit.tighten_bounds())
        self.assertEqual(graphtage.Range(3, 3), edit.bounds())

    def test_the_lattice_is_not_built_until_it_is_read(self):
        """Constructing the edit must not build the lattice; reading the property must, exactly once."""
        edit = self.edit("/usr/local/bin", "/usr/lib/bin")
        self.assertIsNone(edit._edit_distance)
        lattice = edit.edit_distance
        self.assertIsNotNone(edit._edit_distance)
        self.assertIs(lattice, edit.edit_distance)

    def test_the_lattice_agrees_with_the_reported_cost(self):
        """The script a formatter renders has to add up to the cost the edit reported to the matcher."""
        for from_str, to_str in (
                ("/usr/local/bin", "/usr/lib/bin"),
                ("kitten", "sitting"),
                ("aabc", "bcb"),
                ("hello", "hellp"),
                ("", "abc"),
                ("abc", ""),
                ("abcd", "abcd"),
                (b"he\x00o", b"he\xffo"),
        ):
            with self.subTest(from_str=from_str, to_str=to_str):
                edit = self.edit(from_str, to_str)
                lattice = edit.edit_distance
                while lattice.tighten_bounds():
                    pass
                self.assertEqual(edit.bounds().upper_bound, lattice.bounds().upper_bound)

    def test_printing_a_string_node_builds_no_lattice(self):
        """``print_StringNode`` constructs two ``StringEdit``s per node only to ask whether it is quoted."""
        out_stream = StringIO()
        with StringEditDistanceCounter() as counter:
            graphtage.StringNode("/usr/local/bin").print(Printer(ansi_color=False, out_stream=out_stream))
        self.assertEqual(0, counter.count)
        self.assertEqual('"/usr/local/bin"', out_stream.getvalue())

    def test_a_diff_builds_a_lattice_only_for_what_it_renders(self):
        """A dense N-by-N match costs N*N pairs of key/value nodes but renders at most N of them.

        No key survives, so the bipartite matcher prices every one of the 64 candidate pairs, each of which
        holds a key string and a value string. Eagerly building a lattice per candidate is what made a diff of
        a few dozen strings take seconds. The bound here is deliberately loose; it only has to stay far below
        the 128 lattices that the candidates would otherwise account for.

        """
        size = 8
        from_obj = {f"from-key-{i}": f"/usr/local/share/value-{i}" for i in range(size)}
        to_obj = {f"to-key-{i}": f"/opt/local/share/value-{i * 3}" for i in range(size)}
        out_stream = StringIO()
        with StringEditDistanceCounter() as counter:
            diffed = graphtage.json.build_tree(from_obj).diff(graphtage.json.build_tree(to_obj))
            graphtage.json.JSONFormatter.DEFAULT_INSTANCE.print(
                Printer(ansi_color=False, quiet=True, out_stream=out_stream), diffed
            )
        self.assertLessEqual(counter.count, 4 * size)

    def test_a_sixty_key_dict_diff_is_fast(self):
        """Guards the whole point of computing string edit costs without the lattice.

        Every key and every value differs, so the bipartite matcher costs 3600 pairs of key/value nodes, and
        therefore 7200 pairs of strings. Building a lattice for each of those takes the better part of a
        minute; deriving the cost arithmetically takes under a second. The limit is generous so that a slow
        runner cannot flake it.

        """
        rng = random.Random(60)

        def words(count: int, length: int) -> list[str]:
            return [''.join(rng.choices('abcdefghijklmnopqrstuvwxyz', k=length)) for _ in range(count)]

        from_obj = dict(zip(words(60, 12), words(60, 40), strict=True))
        to_obj = dict(zip(words(60, 12), words(60, 40), strict=True))
        with run_with_time_limit(seconds=30):
            edit = graphtage.json.build_tree(from_obj).edits(graphtage.json.build_tree(to_obj))
            while edit.tighten_bounds():
                pass
        self.assertTrue(edit.bounds().definitive())
