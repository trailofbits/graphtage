"""A :class:`graphtage.Filetype` for parsing, diffing, and rendering `flame graphs`_.

There are many libraries in different languages to produce a flame graph from a profiling run.
There unfortunately isn't a standardized textual file format to represent flame graphs.
Graphtage uses this common format:

.. code-block:: none

    function1 #samples
    function1;function2 #samples
    function1;function2;function3 #samples

In other words, each line of the file is a stack trace represented by a ``;``-delimited list of function names
followed by a space and the integer number of times that stack trace was sampled in the profiling run.

.. _flame graphs:
    https://www.brendangregg.com/flamegraphs.html
"""

from typing import Iterable, List, Optional, Union, Iterator

from . import Printer
from .edits import AbstractCompoundEdit, Match, Range, Replace
from .graphtage import (
    BuildOptions, ContainerNode, Edit, Filetype, IntegerNode, ListNode, MultiSetNode, StringNode, TreeNode
)
from .tree import GraphtageFormatter
from .sequences import SequenceFormatter


class FlameGraphParseError(ValueError):
    pass


class Samples(IntegerNode):
    def __init__(self, num_samples: int, total_samples: int):
        if total_samples <= 0:
            raise ValueError("total_samples must be a positive integer")
        elif num_samples < 0:
            raise ValueError("num_samples must be non-negative")
        super().__init__(num_samples)
        self.total_samples: int = total_samples

    @property
    def num_samples(self) -> int:
        return self.object

    @property
    def percent(self) -> float:
        return self.num_samples / self.total_samples

    def calculate_total_size(self) -> int:
        return int(self.percent * 100.0 + 0.5)


class StackTrace(ContainerNode):
    """A stack trace and sample count"""

    def __init__(
            self,
            functions: Iterable[StringNode],
            samples: IntegerNode,
            allow_list_edits: bool = True,
            allow_list_edits_when_same_length: bool = True
    ):
        """Initializes a stack trace.

        Args:
            functions: the functions in the stack trace, in order.
            samples: the number of times this stack trace was sampled in the profiling run.
        """
        if samples.object < 0:
            raise ValueError(f"Invalid number of samples: {samples.object}; the sample count must be non-negative")
        self.functions: ListNode[StringNode] = ListNode(
            functions, allow_list_edits, allow_list_edits_when_same_length
        )
        self.samples: IntegerNode = samples

    def calculate_total_size(self) -> int:
        return self.functions.calculate_total_size() + self.samples.calculate_total_size()

    def print(self, printer: Printer):
        StackTraceFormatter.DEFAULT_INSTANCE.print(printer, self)

    def __eq__(self, other):
        """Two stack traces are the same if their functions exactly match (regardless of their sample count)"""
        return isinstance(other, StackTrace) and self.functions == other.functions

    def __hash__(self):
        return hash(self.functions)

    def __iter__(self):
        yield self.functions
        yield self.samples

    def __len__(self) -> int:
        return 2

    def to_obj(self):
        return self.functions.to_obj() + [self.samples.to_obj()]

    def edits(self, node: TreeNode) -> Edit:
        if self == node and self.samples == node.samples:
            return Match(self, node, cost=0)
        elif isinstance(node, StackTrace):
            return StackTraceEdit(from_node=self, to_node=node)
        else:
            return Replace(self, node)

    def __str__(self):
        return f"{';'.join((str(f.object) for f in self.functions))} {self.samples.object!s}"

    def __repr__(self):
        return f"{self.__class__.__name__}({self.functions!r}, {self.samples!r})"


class StackTraceEdit(AbstractCompoundEdit):
    """An edit on a stack trace."""

    def __init__(self, from_node: "StackTrace", to_node: "StackTrace"):
        """Initializes a stack trace edit.

        Args:
            from_node: The node being edited.
            to_node: The node to which :obj:`from_node` will be transformed.
        """
        self.functions_edit = from_node.functions.edits(to_node.functions)
        self.samples_edit = from_node.samples.edits(to_node.samples)
        super().__init__(from_node, to_node)

    def print(self, formatter: GraphtageFormatter, printer: Printer):
        formatter.get_formatter(self.from_node)(printer, self.from_node)

    def bounds(self) -> Range:
        return self.functions_edit.bounds() + self.samples_edit.bounds()

    def edits(self) -> Iterator[Edit]:
        yield self.functions_edit
        yield self.samples_edit


class FlameGraph(MultiSetNode[StackTrace]):
    pass


class StackTraceFormatter(SequenceFormatter):
    is_partial = True

    def __init__(self):
        super().__init__('', '', ';')

    def item_newline(self, printer: Printer, is_first: bool = False, is_last: bool = False):
        pass

    def print(self, printer: Printer, *args, **kwargs):
        # Flamegraphs are not indented
        printer.indent_str = ""
        super().print(printer, *args, **kwargs)

    def print_StackTrace(self, printer: Printer, node: StackTrace):
        self.print_SequenceNode(printer, node.functions)
        printer.write(" ")
        self.print(printer, node.samples)


class FlameGraphFormatter(SequenceFormatter):
    sub_format_types = [StackTraceFormatter]

    def __init__(self):
        super().__init__('', '', '')

    def print(self, printer: Printer, *args, **kwargs):
        # Flamegraphs are not indented
        printer.indent_str = ""
        super().print(printer, *args, **kwargs)


class FlameGraphFile(Filetype):
    """A textual representation of a flame graph."""
    def __init__(self):
        """Initializes the FlameGraph file type.

        There is no official MIME type associated with a flame graph. Graphtage assigns it the MIME type
        ``text/x-flame-graph``.

        """
        super().__init__(
            'flamegraph',
            'text/x-flame-graph'
        )

    def build_tree(self, path: str, options: Optional[BuildOptions] = None) -> FlameGraph:
        traces: List[StackTrace] = []
        allow_list_edits = options is None or options.allow_list_edits
        allow_list_edits_when_same_length = options is None or options.allow_list_edits_when_same_length
        with open(path, "r") as f:
            for n, line in enumerate(f):
                line = line.strip()
                if not line:
                    continue
                # first parse the int off the end:
                final_int = ""
                while ord('0') <= ord(line[-1]) <= ord('9'):
                    final_int = f"{line[-1]}{final_int}"
                    line = line[:-1]
                    if not line:
                        break
                if not final_int:
                    raise FlameGraphParseError(f"{path}:{n+1} expected the line to end with an integer number of "
                                               "samples")
                samples = int(final_int)
                functions = line.strip().split(";")
                traces.append(StackTrace(
                    functions=(StringNode(f, quoted=False) for f in functions),
                    samples=IntegerNode(samples),
                    allow_list_edits=allow_list_edits,
                    allow_list_edits_when_same_length=allow_list_edits_when_same_length
                ))
        return FlameGraph(traces)

    def build_tree_handling_errors(self, path: str, options: Optional[BuildOptions] = None) -> Union[str, TreeNode]:
        try:
            return self.build_tree(path=path, options=options)
        except FlameGraphParseError as e:
            return str(e)

    def get_default_formatter(self) -> FlameGraphFormatter:
        return FlameGraphFormatter.DEFAULT_INSTANCE
