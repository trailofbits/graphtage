"""
Generic node types for representing abstract syntax trees.
"""
from colorama import Fore

from . import KeyValuePairNode, ListNode, Printer, TreeNode, DictNode, StringNode
from .dataclasses import DataClassNode
from .sequences import SequenceFormatter


class KeywordArgument(KeyValuePairNode):
    pass


class Module(ListNode):
    def print(self, printer: Printer):
        SequenceFormatter('', '', '\n').print(printer, self)


class Assignment(DataClassNode):
    """A node representing an assignment."""

    targets: ListNode
    value: TreeNode

    def print(self, printer: Printer):
        """Prints this node."""
        SequenceFormatter('', '', ', ').print(printer, self.targets)
        with printer.bright():
            printer.write(" = ")
        self.value.print(printer)

    def __str__(self):
        return f"{', '.join(map(str, self.targets.children()))} = {self.value!s}"


class CallArguments(ListNode):
    pass


class CallKeywords(DictNode):
    pass


class Call(DataClassNode):
    """A node representing a function call."""

    func: TreeNode
    args: CallArguments
    kwargs: CallKeywords

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        if isinstance(self.func, StringNode):
            self.func.quoted = False

    def print(self, printer: Printer):
        with printer.color(Fore.YELLOW):
            self.func.print(printer)
        printer.write("(")
        SequenceFormatter('', '', ', ').print(printer, self.args)
        if self.args and len(self.kwargs) > 0:
            printer.write(", ")
        for kvp in self.kwargs:
            with printer.color(Fore.RED):
                kvp.key.print(printer)
            with printer.bright():
                printer.write("=")
            kvp.value.print(printer)
        printer.write(")")

    def __str__(self):
        args = ", ".join([str(a) for a in self.args] + [
            f"{kvp.key!s}={kvp.value!s}"
            for kvp in self.kwargs
        ])
        return f"{self.func!s}({args})"


class Subscript(DataClassNode):
    """A node representing an object subscript (i.e., the `[]` operator)"""

    value: TreeNode
    slice: TreeNode

    def print(self, printer: Printer):
        self.value.print(printer)
        with printer.color(Fore.LIGHTBLUE_EX):
            printer.write("[")
        self.slice.write(printer)
        with printer.color(Fore.LIGHTBLUE_EX):
            printer.write("]")


class Import(DataClassNode):
    names: ListNode
    from_name: StringNode

    def __init__(self, names: ListNode, from_name: StringNode):
        super().__init__(names=names, from_name=from_name)
        self.from_name.quoted = False
        for child in self.names:
            if isinstance(child, StringNode):
                child.quoted = False

    def print(self, printer: Printer):
        if self.from_name.object:
            with printer.color(Fore.YELLOW):
                printer.write("from ")
            self.from_name.print(printer)
            printer.write(" ")
        with printer.color(Fore.YELLOW):
            printer.write("import ")
        SequenceFormatter('', '', ', ').print(printer, self.names)
