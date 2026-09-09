from typing import Any
from unittest import TestCase

import graphtage
from graphtage import expressions
from graphtage.constraints import ConditionalMatcher, MatchIf, MatchUnless
from graphtage.edits import Edit
from graphtage.json import build_tree


def diff_with_constraint(matcher: type[ConditionalMatcher], expression: str, from_obj: Any, to_obj: Any) -> Edit:
    """Diffs two Python objects with a matcher applied to every node of the first tree.

    This mirrors what ``graphtage.__main__`` does for ``--match-if`` and ``--match-unless``: the constraint is
    attached to every node reached by a depth-first search, not only to the root.

    Args:
        matcher: the :class:`ConditionalMatcher` subclass to apply.
        expression: the constraint expression, in the syntax accepted by :mod:`graphtage.expressions`.
        from_obj: the Python object to build the "from" tree from.
        to_obj: the Python object to build the "to" tree from.

    Returns:
        Edit: the edit produced for the roots of the two trees.
    """
    condition = expressions.parse(expression)
    from_tree = build_tree(from_obj)
    for node in from_tree.dfs():
        matcher.apply(node, condition)
    return from_tree.diff(build_tree(to_obj)).edit


class TestConstraints(TestCase):
    """Tests the ``--match-if`` and ``--match-unless`` node pairing constraints."""

    def test_match_if_permits_pair_when_condition_holds(self):
        """`--match-if` must pair two dicts whose compared values are equal.

        This is the example from the ``--match-if`` help text. It previously failed because ``MatchIf`` bound the
        raw :class:`graphtage.TreeNode` objects, so ``from['foo']`` was a ``KeyValuePairNode`` carrying its key and
        therefore never equalled ``to['bar']``.
        """
        edit = diff_with_constraint(
            MatchIf,
            "from['foo'] == to['bar']",
            {"foo": "same", "other": 1},
            {"bar": "same", "other": 2},
        )
        self.assertIsInstance(edit, graphtage.MultiSetEdit)

    def test_match_if_refuses_pair_when_condition_fails(self):
        """`--match-if` must still refuse a pair whose compared values differ."""
        edit = diff_with_constraint(
            MatchIf,
            "from['foo'] == to['bar']",
            {"foo": "same", "other": 1},
            {"bar": "different", "other": 2},
        )
        self.assertIsInstance(edit, graphtage.Replace)

    def test_match_if_permits_pair_when_expression_raises(self):
        """A `--match-if` expression that raises must leave the pair unconstrained.

        The constraint is applied to every node, including leaves, where subscripting raises. ``MatchIf`` previously
        answered such an error with a ``Replace``, which refused every leaf and collapsed the diff even once the
        operands were bound correctly.
        """
        edit = diff_with_constraint(MatchIf, "from['id'] == to['id']", "hello", "hellp")
        self.assertIsInstance(edit, graphtage.StringEdit)

    def test_match_unless_refuses_pair_when_condition_holds(self):
        """`--match-unless` must refuse a pair whose compared values differ."""
        edit = diff_with_constraint(
            MatchUnless,
            "from['foo'] != to['bar']",
            {"foo": "same", "other": 1},
            {"bar": "different", "other": 2},
        )
        self.assertIsInstance(edit, graphtage.Replace)

    def test_match_unless_permits_pair_when_condition_fails(self):
        """`--match-unless` must pair two dicts whose compared values are equal."""
        edit = diff_with_constraint(
            MatchUnless,
            "from['foo'] != to['bar']",
            {"foo": "same", "other": 1},
            {"bar": "same", "other": 2},
        )
        self.assertIsInstance(edit, graphtage.MultiSetEdit)

    def test_match_unless_permits_pair_when_expression_raises(self):
        """A `--match-unless` expression that raises must leave the pair unconstrained."""
        edit = diff_with_constraint(MatchUnless, "from['id'] != to['id']", "hello", "hellp")
        self.assertIsInstance(edit, graphtage.StringEdit)

    def test_match_if_and_match_unless_agree(self):
        """The two options must produce the same diff for logically equivalent expressions.

        The help text presents ``--match-unless`` as ``--match-if`` with the sense of the test inverted, so the two
        spellings of the same constraint must not disagree.
        """
        from_obj = {"foo": "same", "other": 1}
        to_obj = {"bar": "same", "other": 2}
        match_if = diff_with_constraint(MatchIf, "from['foo'] == to['bar']", from_obj, to_obj)
        match_unless = diff_with_constraint(MatchUnless, "from['foo'] != to['bar']", from_obj, to_obj)
        self.assertIsInstance(match_if, type(match_unless))
