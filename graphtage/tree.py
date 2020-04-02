from abc import abstractmethod, ABCMeta
from typing import Optional

from .printer import Printer


class TreeNode(metaclass=ABCMeta):
    _total_size = None

    @abstractmethod
    def edits(self, node) -> 'graphtage.edits.Edit':
        pass

    @property
    def total_size(self) -> int:
        if self._total_size is None:
            self._total_size = self.calculate_total_size()
        return self._total_size

    @abstractmethod
    def calculate_total_size(self) -> int:
        return 0

    @abstractmethod
    def print(self, printer: Printer, diff: Optional['graphtage.Diff'] = None):
        pass


class ContainerNode(TreeNode, metaclass=ABCMeta):
    pass
