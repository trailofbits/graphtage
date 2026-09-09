import logging
from abc import ABCMeta, abstractmethod

from . import expressions, graphtage
from .edits import Edit

log = logging.getLogger('graphtage')


class ConditionalMatcher(metaclass=ABCMeta):
    """Decides whether two nodes may be paired, based on a commandline expression.

    The expression is evaluated with ``from`` and ``to`` bound to the :meth:`TreeNode.to_obj` representations of the
    two nodes, so it operates on plain Python values rather than on Graphtage's node objects.

    Constraints are attached to every node of a tree, so an expression written for one kind of node will raise on the
    others; for example, subscripting a string by a dictionary key raises a :class:`TypeError`. An expression that
    raises leaves the pair unconstrained rather than refusing it.
    """

    def __init__(self, condition: expressions.Expression):
        self.condition: expressions.Expression = condition

    @abstractmethod
    def __call__(self, from_node: graphtage.TreeNode, to_node: graphtage.TreeNode) -> Edit | None:
        raise NotImplementedError()

    @classmethod
    def apply(cls, node: graphtage.TreeNode, condition: expressions.Expression):
        node.add_edit_modifier(cls(condition))


class MatchIf(ConditionalMatcher):
    """Refuses to pair two nodes unless the condition evaluates to a true value."""

    def __call__(self, from_node: graphtage.TreeNode, to_node: graphtage.TreeNode) -> Edit | None:
        try:
            if self.condition.eval(locals={'from': from_node.to_obj(), 'to': to_node.to_obj()}):
                return None
        except Exception as e:
            log.debug(f"{e!s} while evaluating --match-if for nodes {from_node} and {to_node}")
            return None
        return graphtage.Replace(from_node, to_node)


class MatchUnless(ConditionalMatcher):
    """Refuses to pair two nodes when the condition evaluates to a true value."""

    def __call__(self, from_node: graphtage.TreeNode, to_node: graphtage.TreeNode) -> Edit | None:
        try:
            if self.condition.eval(locals={'from': from_node.to_obj(), 'to': to_node.to_obj()}):
                return graphtage.Replace(from_node, to_node)
        except Exception as e:
            log.debug(f"{e!s} while evaluating --match-unless for nodes {from_node} and {to_node}")
        return None
