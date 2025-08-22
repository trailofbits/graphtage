from unittest import TestCase

import graphtage
from graphtage.constraints import MatchIf, MatchUnless
from graphtage.json import build_tree
from graphtage import expressions


class TestConstraints(TestCase):
    def test_match_if(self):
        expr = expressions.parse("from.key == 'foo' && to.key == 'bar'")
        from_tree = build_tree({
            "foo": [1, 2, 3]
        })
        for node in from_tree.dfs():
            MatchIf.apply(node, expr)
        to_tree = build_tree({
            "bar": [1, 2, 4]
        })
        diff = from_tree.diff(to_tree)
        self.assertIsInstance(diff.edit, graphtage.Replace)

    def test_match_unless(self):
        expr = expressions.parse("from.key == 'foo' && to.key == 'bar'")
        from_tree = build_tree({
            "foo": [1, 2, 3]
        })
        for node in from_tree.dfs():
            MatchUnless.apply(node, expr)
        to_tree = build_tree({
            "bar": [1, 2, 4]
        })
        diff = from_tree.diff(to_tree)
        self.assertIsInstance(diff.edit, graphtage.MultiSetEdit)
