from io import StringIO
from unittest import TestCase

import graphtage
from graphtage.pydiff import build_tree

from graphtage.printer import Printer


class TestPyDiff(TestCase):
    def test_build_tree(self):
        self.assertIsInstance(build_tree([1, 2, 3, 4]), graphtage.ListNode)
        self.assertIsInstance(build_tree({1: 2, 'a': 'b'}), graphtage.DictNode)

    def test_diff(self):
        build_tree([1, 2, 3, 4]).diff(build_tree([1, 2, 3, '4']))
