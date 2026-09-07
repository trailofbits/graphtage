from unittest import TestCase

import graphtage
import graphtage.yaml
from graphtage.utils import Tempfile

RULES = b"""- macro: never_true
  condition: (evt.num=0)

- macro: always_true
  condition: (evt.num>=0)

- macro: spawned_process
  condition: evt.type in (execve, execveat) and evt.dir=<

- rule: ls run
  desc: ls run
  condition: spawned_process and proc.name=ls
  output: ls run
  priority: INFO
  tags: [process]
"""

REORDERED_RULES = b"""- macro: never_true
  condition: (evt.num=0)

- macro: always_true
  condition: (evt.num>=0)

- rule: ls run
  desc: ls run
  condition: spawned_process and proc.name=ls
  output: ls run
  priority: INFO
  tags: [process]

- macro: spawned_process
  condition: evt.type in (execve, execveat) and evt.dir=<
"""


class TestIssue56(TestCase):
    """Reproduces https://github.com/trailofbits/graphtage/issues/56"""

    @staticmethod
    def build(content: bytes, options: graphtage.BuildOptions) -> graphtage.TreeNode:
        with Tempfile(content) as path:
            return graphtage.FILETYPES_BY_TYPENAME["yaml"].build_tree(path, options=options)

    def diff(self, options: graphtage.BuildOptions):
        return self.build(RULES, options).diff(self.build(REORDERED_RULES, options))

    def test_reordered_documents_are_an_edit_by_default(self):
        self.assertGreater(self.diff(graphtage.BuildOptions()).edited_cost(), 0)

    def test_reordered_documents_match_when_ignoring_list_order(self):
        """The two files differ only in where one document sits, so `graphtage` must report no change.

        The second assertion mirrors how `graphtage.__main__` decides its exit status, so this is also an
        assertion that the command exits 0.

        """
        diff = self.diff(graphtage.BuildOptions(ignore_list_order=True))
        self.assertEqual(0, diff.edited_cost())
        self.assertFalse(any(any(e.has_non_zero_cost() for e in n.edit_list) for n in diff.dfs()))
