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

from typing import Iterable, List, Optional, Union

from . import Printer
from .edits import EditSequence
from .graphtage import BuildOptions, Edit, Filetype, IntegerNode, ListNode, MultiSetNode, StringNode, TreeNode
from .sequences import SequenceFormatter
from .tree import GraphtageFormatter


class FlameGraphParseError(ValueError):
    pass


class StackTrace(ListNode[StringNode]):
    def __init__(
            self,
            functions: Iterable[StringNode],
            samples: IntegerNode,
            allow_list_edits: bool = True,
            allow_list_edits_when_same_length: bool = True
    ):
        super().__init__(functions, allow_list_edits, allow_list_edits_when_same_length)
        self.samples: IntegerNode = samples

    def to_obj(self):
        return [n.to_obj() for n in self] + [self.samples]

    def edits(self, node: TreeNode) -> Edit:
        # first, match the functions:
        edit = super().edits(node)
        if not isinstance(node, StackTrace) or self.samples == node.samples:
            return edit
        # now match the samples:
        return EditSequence(
            from_node=self,
            to_node=node,
            edits=(edit, self.samples.edits(node.samples))
        )


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
        self.print_SequenceNode(printer, node)
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
                if not final_int:
                    raise FlameGraphParseError(f"{path}:{n+1} expected the line to end with an integer number of "
                                               "samples")
                samples = int(final_int)
                functions = line.strip().split(";")
                if not functions:
                    raise FlameGraphParseError(f"{path}:{n+1} the line did not contain a stack trace")
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
