import os
from hashlib import sha1
from typing import Optional

from . import Printer
from .edits import Edit, Match
from .graphtage import LeafNode
from .levenshtein import levenshtein_distance
from .search import POSITIVE_INFINITY
from .tree import TreeNode


class File(LeafNode):
    def __init__(self, base_dir: str, path: str):
        self.absolute_path: str = os.path.abspath(path)
        self._filesize: Optional[int] = None
        self._sha1hash: Optional[int] = None
        super().__init__(os.path.relpath(self.absolute_path, os.path.abspath(base_dir)))

    def calculate_total_size(self) -> int:
        return self.filesize

    def __eq__(self, other):
        return isinstance(other, File) and other.path == self.path and hash(other) == hash(self)

    def edits(self, node: TreeNode) -> Edit:
        assert isinstance(node, File)
        if self == node:
            return Match(self, node, 0)
        elif hash(self) == hash(node):
            return FileMoved(self, node)
        elif self.path == node.path:
            return FileEdited(self, node)
        else:
            return Match(self, node, POSITIVE_INFINITY)

    @property
    def filename(self) -> str:
        return os.path.basename(self.path)

    @property
    def filesize(self) -> int:
        if self._filesize is None:
            self._filesize = os.stat(self.path).st_size
        return self._filesize

    @property
    def sha1hash(self) -> int:
        if self._sha1hash is None:
            h = sha1()
            with open(self.path, 'rb') as f:
                h.update(f.read())
            self._sha1hash = int.from_bytes(h.digest(), byteorder='big', signed=False )
        return self._sha1hash

    def __hash__(self):
        return self.sha1hash

    @property
    def path(self) -> str:
        return self.object

    @property
    def directory(self) -> str:
        return os.path.dirname(self.path)


class FileMoved(Edit):
    def __init__(self, file_before: File, file_after: File):
        dir1, dir2 = file_before.directory, file_after.directory
        if dir1 == dir2:
            move_cost = 0
        else:
            common_path = os.path.commonpath([dir1, dir2])
            move_cost = dir1[len(common_path):].count('/') + dir2[len(common_path):].count('/')
        cost = levenshtein_distance(file_before.filename, file_after.filename) + move_cost
        super().__init__(
            from_node=file_before,
            to_node=file_after,
            constant_cost=cost,
            cost_upper_bound=cost
        )

    def print(self, printer: Printer):
        self.from_node.print(printer)


class FileEdited(Edit):
    def __init__(self, file_before: File, file_after: File):
        cost = max(file_before.filesize, file_after.filesize)
        super().__init__(
            from_node=file_before,
            to_node=file_after,
            constant_cost=cost,
            cost_upper_bound=cost
        )

    def print(self, printer: Printer):
        self.from_node.print(printer)
