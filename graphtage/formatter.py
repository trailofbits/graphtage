import inspect
import logging
from abc import ABCMeta
from typing import Any, Callable, List, Optional, Sequence, Type, TypeVar

from .printer import Printer
from .tree import EditedTreeNode, TreeNode


log = logging.getLogger(__name__)


FORMATTERS: Sequence['Formatter'] = []


class FormatterChecker(ABCMeta):
    def __init__(cls, name, bases, clsdict):
        if len(cls.mro()) > 2 and not cls.__abstractmethods__:
            # Instantiate a version of the Formatter to add it to our global dicts:
            # If the formatter has a custom init, we can't use it as a default
            try:
                if hasattr(cls, '__init__') and inspect.signature(cls.__init__).bind(None):
                    pass
                # if we got this far then the Formatter class has a default constructor
                instance = cls()
                for member, member_type in inspect.getmembers(instance):
                    if member.startswith('print_'):
                        sig = inspect.signature(member_type)
                        if 'printer' in sig.parameters:
                            a = sig.parameters['printer'].annotation
                            if a is not None and not issubclass(Printer, a):
                                raise TypeError(f"The type annotation for {name}.{member}(printer: {a}) was expected to be a superclass of graphtage.printer.Printer")
                        if 'node' in sig.parameters:
                            a = sig.parameters['node'].annotation
                            if a is not None and not issubclass(a, TreeNode):
                                raise TypeError(f"The type annotation for {name}.{member}(node: {a}) was expected to be a subclass of graphtage.tree.TreeNode")
                        if 'with_edits' in sig.parameters:
                            a = sig.parameters['with_edits'].annotation
                            if a is not None and a is not bool:
                                raise TypeError(f"The type annotation for {name}.{member}(with_edits: {a}) was expected to be a bool")
                if not instance.is_partial:
                    FORMATTERS.append(instance)
                setattr(cls, 'DEFAULT_INSTANCE', instance)
            except TypeError:
                log.debug(f"Formatter {name} cannot be instantiated as a default constructor")
        super().__init__(name, bases, clsdict)


T = TypeVar('T', bound=TreeNode)


def get_formatter(node_type: Type[T], base_formatter: Optional['Formatter'] = None) -> Optional[Callable[[Printer, T], Any]]:
    if base_formatter is not None:
        base_formatters = frozenset([base_formatter.__class__] + [s.__class__ for s in base_formatter.sub_formatters])
    else:
        base_formatters = frozenset()
    if base_formatter is not None:
        for c in node_type.mro():
            if hasattr(base_formatter, f'print_{c.__name__}'):
                return getattr(base_formatter, f'print_{c.__name__}')
            for sub_formatter in base_formatter.sub_formatters:
                if hasattr(sub_formatter, f'print_{c.__name__}'):
                    return getattr(sub_formatter, f'print_{c.__name__}')
    for c in node_type.mro():
        for formatter in FORMATTERS:
            if formatter.__class__ not in base_formatters and hasattr(formatter, f'print_{c.__name__}'):
                return getattr(formatter, f'print_{c.__name__}')
    return None


class Formatter(metaclass=FormatterChecker):
    DEFAULT_INSTANCE: __qualname__ = None
    sub_format_types: Sequence[Type['Formatter']] = ()
    sub_formatters: List['Formatter'] = []
    parent: Optional['Formatter'] = None
    is_partial: bool = False

    @property
    def root(self) -> 'Formatter':
        node = self
        while node.parent is not None:
            node = node.parent
        return node

    def __new__(cls, *args, **kwargs):
        ret: Formatter = super().__new__(cls, *args, **kwargs)
        setattr(ret, 'sub_formatters', [])
        for sub_formatter in ret.sub_format_types:
            ret.sub_formatters.append(sub_formatter())
            ret.sub_formatters[-1].parent = ret
        return ret

    def get_formatter(self, node: T) -> Callable[[Printer, T], Any]:
        return get_formatter(node.__class__, base_formatter=self)

    def print(self, printer: Printer, node: TreeNode, with_edits: bool = True):
        if self.parent is not None:
            return self.root.print(printer, node, with_edits)
        formatter = self.get_formatter(node)
        if formatter is not None:
            if with_edits and isinstance(node, EditedTreeNode) and node.edit is not None:
                node.edit.print(self, printer)
            else:
                formatter(printer=printer, node=node)
        else:
            log.debug(f"""There is no formatter that can handle nodes of type {node.__class__.__name__}
Falling back to the node's internal printer
Registered formatters: {''.join([f.__class__.__name__ for f in FORMATTERS])}""")
            node.print(printer)
