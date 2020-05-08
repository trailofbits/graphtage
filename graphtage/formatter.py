import inspect
import logging
from abc import ABCMeta, abstractmethod
from typing import Any, Callable, Generic, List, Optional, Sequence, Set, Type, TypeVar

from .printer import Printer


log = logging.getLogger(__name__)


FORMATTERS: Sequence['Formatter[Any]'] = []


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
            except TypeError:
                log.debug(f"Formatter {name} cannot be instantiated as a default constructor")
                instance = None
            if instance is not None:
                for member, member_type in inspect.getmembers(instance):
                    if member.startswith('print_'):
                        sig = inspect.signature(member_type)
                        if 'printer' in sig.parameters:
                            a = sig.parameters['printer'].annotation
                            if a is not None and a != inspect.Signature.empty and inspect.isclass(a) \
                                    and not issubclass(Printer, a):
                                raise TypeError(f"The type annotation for {name}.{member}(printer: {a}) was expected to be a superclass of graphtage.printer.Printer")
                if not instance.is_partial:
                    FORMATTERS.append(instance)
                setattr(cls, 'DEFAULT_INSTANCE', instance)
        super().__init__(name, bases, clsdict)


T = TypeVar('T')


def _get_formatter(
        node_type: Type[T],
        base_formatter: 'Formatter',
        tested: Set[Type['Formatter']]
) -> Optional[Callable[[Printer, T], Any]]:
    if base_formatter.__class__ not in tested:
        grandchildren = []
        for c in node_type.mro():
            if hasattr(base_formatter, f'print_{c.__name__}'):
                return getattr(base_formatter, f'print_{c.__name__}')
            for sub_formatter in base_formatter.sub_formatters:
                if sub_formatter not in tested:
                    if hasattr(sub_formatter, f'print_{c.__name__}'):
                        return getattr(sub_formatter, f'print_{c.__name__}')
                    grandchildren.extend(sub_formatter.sub_formatters)
        tested.add(base_formatter.__class__)
        tested |= set(s.__class__ for s in base_formatter.sub_formatters)
        for grandchild in grandchildren:
            ret = _get_formatter(node_type, grandchild, tested)
            if ret is not None:
                return ret
    if base_formatter.parent is not None:
        return _get_formatter(node_type, base_formatter.parent, tested)


def get_formatter(
        node_type: Type[T],
        base_formatter: Optional['Formatter'] = None
) -> Optional[Callable[[Printer, T], Any]]:
    tested_formatters: Set[Type[Formatter]] = set()
    if base_formatter is not None:
        ret = _get_formatter(node_type, base_formatter, tested_formatters)
        if ret is not None:
            return ret
    for formatter in FORMATTERS:
        if formatter.__class__ not in tested_formatters:
            ret = _get_formatter(node_type, formatter, tested_formatters)
            if ret is not None:
                return ret
    return None


class Formatter(Generic[T], metaclass=FormatterChecker):
    DEFAULT_INSTANCE: 'Formatter[T]' = None
    sub_format_types: Sequence[Type['Formatter[T]']] = ()
    sub_formatters: List['Formatter[T]'] = []
    parent: Optional['Formatter[T]'] = None
    is_partial: bool = False

    @property
    def root(self) -> 'Formatter[T]':
        node = self
        while node.parent is not None:
            node = node.parent
        return node

    def __new__(cls, *args, **kwargs):
        ret: Formatter[T] = super().__new__(cls, *args, **kwargs)
        setattr(ret, 'sub_formatters', [])
        for sub_formatter in ret.sub_format_types:
            ret.sub_formatters.append(sub_formatter())
            ret.sub_formatters[-1].parent = ret
        return ret

    def get_formatter(self, item: T) -> Optional[Callable[[Printer, T], Any]]:
        return get_formatter(item.__class__, base_formatter=self)

    @abstractmethod
    def print(self, printer: Printer, item: T):
        raise NotImplementedError()
