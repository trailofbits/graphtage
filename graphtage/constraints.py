from abc import ABCMeta, abstractmethod
import logging
from typing import Optional

from .edits import Edit
from . import expressions
from . import graphtage

log = logging.getLogger('graphtage')


class ConditionalMatcher(metaclass=ABCMeta):
    def __init__(self, condition: expressions.Expression):
        self.condition: expressions.Expression = condition

    @abstractmethod
    def __call__(self, from_node: graphtage.TreeNode, to_node: graphtage.TreeNode) -> Optional[Edit]:
        raise NotImplementedError()

    @classmethod
    def apply(cls, node: graphtage.TreeNode, condition: expressions.Expression):
        node.add_edit_modifier(cls(condition))


class MatchIf(ConditionalMatcher):
    def __call__(self, from_node: graphtage.TreeNode, to_node: graphtage.TreeNode) -> Optional[Edit]:
        try:
            if self.condition.eval(locals={'from': from_node, 'to': to_node}):
                return None
        except Exception as e:
            log.debug(f"{e!s} while evaluating --match-if for nodes {from_node} and {to_node}")
        return graphtage.Replace(from_node, to_node)


class MatchUnless(ConditionalMatcher):
    def __call__(self, from_node: graphtage.TreeNode, to_node: graphtage.TreeNode) -> Optional[Edit]:
        try:
            if self.condition.eval(locals={'from': from_node.to_obj(), 'to': to_node.to_obj()}):
                return graphtage.Replace(from_node, to_node)
        except Exception as e:
            log.debug(f"{e!s} while evaluating --match-unless for nodes {from_node} and {to_node}")
        return None
