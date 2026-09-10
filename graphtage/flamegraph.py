"""A :class:`graphtage.Filetype` for parsing, diffing, and rendering `flame graphs`_.

Many libraries in many languages produce a flame graph from a profiling run, but there is no
standardized textual file format to represent one. Graphtage reads the "folded stacks" format that
`stackcollapse-perf.pl`_ and its siblings emit:

.. code-block:: none

    function1 12
    function1;function2 34
    function1;function2;function3 56

Each line is one stack trace, written as a ``;``-delimited list of function names, followed by a
space and the integer number of times that stack trace was sampled during the profiling run.

Graphtage models a flame graph as a mapping from stack trace to sample count, so two profiles of the
same program match their shared stack traces directly and only the stack traces unique to one
profile are compared against each other.

.. _flame graphs:
    https://www.brendangregg.com/flamegraphs.html

.. _stackcollapse-perf.pl:
    https://github.com/brendangregg/FlameGraph

"""

import os

from .graphtage import (
    BuildOptions,
    DictNode,
    Filetype,
    IntegerNode,
    KeyValuePairNode,
    ListNode,
    StringNode,
)
from .printer import Printer
from .sequences import SequenceFormatter
from .tree import GraphtageFormatter


class FlameGraphParseError(ValueError):
    """Raised when a file cannot be parsed as folded stacks."""


class StackFrames(ListNode[StringNode]):
    """The function names of a single stack trace, outermost first."""


class StackTrace(KeyValuePairNode):
    """A single stack trace and the number of times it was sampled.

    The frames are the key and the sample count is the value, so two flame graphs match the stack
    traces they have in common without costing an edit for each one.

    """

    def __init__(self, frames: StackFrames, samples: IntegerNode, allow_frame_edits: bool = True):
        """Initializes a stack trace.

        Args:
            frames: The functions in the stack trace, outermost first.
            samples: The number of times this stack trace was sampled in the profiling run.
            allow_frame_edits: If :const:`False`, only consider matching this stack trace against one whose frames
                are identical.

        Raises:
            ValueError: If the sample count is negative.

        """
        if samples.object < 0:
            raise ValueError(f"Invalid number of samples: {samples.object}; the sample count must be non-negative")
        super().__init__(key=frames, value=samples, allow_key_edits=allow_frame_edits)

    @property
    def frames(self) -> StackFrames:
        """The functions in this stack trace, outermost first."""
        return self.key

    @property
    def samples(self) -> IntegerNode:
        """The number of times this stack trace was sampled."""
        return self.value

    def to_obj(self):
        return ';'.join(self.frames.to_obj()), self.samples.to_obj()


class FlameGraph(DictNode):
    """A flame graph: a mapping from stack trace to sample count."""

    def to_obj(self):
        return dict(trace.to_obj() for trace in self)


def build_tree(path: str, options: BuildOptions | None = None) -> FlameGraph:
    """Constructs a :class:`FlameGraph` from a file of folded stacks.

    Args:
        path: The path to the file to be parsed.
        options: An optional set of options for building the tree.

    Returns:
        FlameGraph: The resulting flame graph.

    Raises:
        FlameGraphParseError: If a line is not a stack trace followed by a sample count.

    """
    if options is None:
        options = BuildOptions()
    traces: list[StackTrace] = []
    with open(path, encoding="utf-8") as f:
        for line_number, raw_line in enumerate(f, start=1):
            line = raw_line.strip()
            if not line:
                continue
            stack, separator, count = line.rpartition(" ")
            if not separator:
                raise FlameGraphParseError(
                    f"{path}:{line_number}: expected a stack trace followed by a space and a sample count"
                )
            try:
                num_samples = int(count)
            except ValueError:
                raise FlameGraphParseError(
                    f"{path}:{line_number}: expected the line to end with an integer sample count, not {count!r}"
                )
            if num_samples < 0:
                raise FlameGraphParseError(f"{path}:{line_number}: the sample count must be non-negative")
            traces.append(StackTrace(
                frames=StackFrames(
                    (StringNode(frame, quoted=False) for frame in stack.split(";")),
                    options.allow_list_edits,
                    options.allow_list_edits_when_same_length
                ),
                samples=IntegerNode(num_samples),
                allow_frame_edits=options.allow_key_edits
            ))
    return FlameGraph(traces, auto_match_keys=options.auto_match_keys)


class StackFramesFormatter(SequenceFormatter):
    """A formatter for the frames of a single stack trace."""
    is_partial = True

    def __init__(self):
        """Initializes the formatter.

        Equivalent to::

            super().__init__('', '', ';')

        """
        super().__init__('', '', ';')

    def print_StackFrames(self, *args, **kwargs):
        """Prints the frames of a stack trace.

        Equivalent to::

            super().print_SequenceNode(*args, **kwargs)

        """
        super().print_SequenceNode(*args, **kwargs)

    def item_newline(self, printer: Printer, is_first: bool = False, is_last: bool = False):
        """An empty implementation, since a stack trace is printed on a single line."""
        pass

    def items_indent(self, printer: Printer):
        """Returns :obj:`printer` because stack frames are not indented."""
        return printer


class FlameGraphStackFormatter(SequenceFormatter):
    """A formatter for the sequence of stack traces in a flame graph."""
    is_partial = True

    sub_format_types = (StackFramesFormatter,)

    def __init__(self):
        """Initializes the formatter.

        Equivalent to::

            super().__init__('', '', '')

        """
        super().__init__('', '', '')

    def print_FlameGraph(self, *args, **kwargs):
        """Prints a flame graph.

        Equivalent to::

            super().print_SequenceNode(*args, **kwargs)

        """
        super().print_SequenceNode(*args, **kwargs)

    def print_StackTrace(self, printer: Printer, node: StackTrace):
        """Prints one folded stack: the frames, a space, and the sample count."""
        self.print(printer, node.frames)
        printer.write(" ")
        self.print(printer, node.samples)

    def item_newline(self, printer: Printer, is_first: bool = False, is_last: bool = False):
        """Prints a newline on all but the first and last stack traces."""
        if not is_first and not is_last:
            printer.newline()

    def items_indent(self, printer: Printer):
        """Returns :obj:`printer` because stack traces are not indented."""
        return printer


class FlameGraphFormatter(GraphtageFormatter):
    """Top-level formatter for flame graphs."""
    sub_format_types = (FlameGraphStackFormatter,)


class FlameGraphFile(Filetype):
    """The flame graph filetype."""

    def __init__(self):
        """Initializes the flame graph filetype.

        There is no official MIME type for a flame graph, so Graphtage assigns it ``text/x-flame-graph``.

        """
        super().__init__(
            'flamegraph',
            'text/x-flame-graph'
        )

    def build_tree(self, path: str, options: BuildOptions | None = None) -> FlameGraph:
        """Equivalent to :func:`build_tree`"""
        return build_tree(path, options=options)

    def build_tree_handling_errors(self, path: str, options: BuildOptions | None = None) -> str | FlameGraph:
        try:
            return self.build_tree(path=path, options=options)
        except (FlameGraphParseError, OSError, UnicodeDecodeError) as e:
            return f"Error parsing {os.path.basename(path)}: {e}"

    def get_default_formatter(self) -> FlameGraphFormatter:
        return FlameGraphFormatter.DEFAULT_INSTANCE
