"""A module for extensible and reusable textual formatting.

Why does the formatter module exist?
====================================

This module is completely generic, with no ties to Graphtage. However, it is easiest to see why it is necessary with an
example from Graphtage: filetypes, edits, and nodes. The problem is that Graphtage is designed to be capable of use as a
library to define new filetypes, node types, and edits. Graphtage also allows *any* input type to be output as if it
were any other type. For example, two JSON files could be diffed and the output printed in YAML. Or a JSON file could be
diffed against *another* JSON file and then output in some other format. This is enabled through use of an intermediate
representation based on :class:`graphtage.TreeNode`. Say a developer uses Graphtage as a library to develop support for
some new file format as both input and output. The intermediate representation means that we would immediately be able
to compare files in that new format to both JSON and YAML files. However, what if the user requests that a JSON file
be output as the new format? Or what if an input file in the new format is to be output as YAML? How does the
preexisting YAML formatter know how to deal with the new node and edit types defined for the new format?

.. _What the formatter module can do:

What the formatter module can do
================================

This is where the formatter module comes into play. It uses Python magic and wizardry, and a bit of dynamic type
inference, to figure out the best formatter for a specific object. The following examples should make this more clear.

Examples:

    >>> from graphtage.printer import Printer
    >>> from graphtage.formatter import BasicFormatter, get_formatter
    >>> class StringFormatter(BasicFormatter[str]):
    ...     def print_str(self, printer: Printer, item: str):
    ...         printer.write(f"StringFormatter: {item}")
    ...         printer.newline()
    ...
    >>> get_formatter(str)(Printer(), "foo")
    StringFormatter: foo

    The first thing to note here is that simply subclassing :class:`Formatter` will register it with
    this module so it will be considered when resolving the best formatter for an object in calls to
    :func:`get_formatter`. This registration can be disabled by setting the class variable :attr:`Formatter.is_partial`
    to :const:`True`.

    >>> class IntFormatter(BasicFormatter[int]):
    ...     def print_int(self, printer: Printer, item: int):
    ...         printer.write(f"IntFormatter: {item}")
    ...         printer.newline()
    ...
    >>> get_formatter(int)(Printer(), 1337)
    IntFormatter: 1337

    It works for any object type. It is not necessary to specify a generic type when subclassing :class:`Formatter` or
    :class:`BasicFormatter`; this is just available for convenience, readability, and automated type checking.

    The next thing we will demonstrate is how formatter lookup works with inheritance:

    >>> class Foo:
    ...     pass
    ...
    >>> class FooFormatter(BasicFormatter[Foo]):
    ...     def print_Foo(self, printer: Printer, item: Foo):
    ...         printer.write("FooFormatter")
    ...         printer.newline()
    ...
    >>> get_formatter(Foo)(Printer(), Foo())
    FooFormatter
    >>> class Bar(Foo):
    ...     def __init__(self, bar):
    ...         self.bar = bar
    ...
    >>> get_formatter(Bar)(Printer(), Bar(None))
    FooFormatter

    Straightforward enough. But what if we define a separate formatter that handles objects of type ``Bar``?

    >>> class BarFormatter(BasicFormatter[Bar]):
    ...     def print_Bar(self, printer: Printer, item: Bar):
    ...         printer.write("BarFormatter: ")
    ...         self.print(printer, item.bar)
    ...
    >>> get_formatter(Bar)(Printer(), Bar(None))
    BarFormatter: None
    >>> get_formatter(Bar)(Printer(), Bar(Foo()))
    BarFormatter: FooFormatter
    >>> get_formatter(Bar)(Printer(), Bar(Bar("foo")))
    BarFormatter: BarFormatter: StringFormatter: foo
    >>> get_formatter(Bar)(Printer(), Bar(Bar(1337)))
    BarFormatter: BarFormatter: IntFormatter: 1337

    Cool, huh? But what if there are collisions? Let's extend ``BarFormatter`` to also handle strings:

    .. code-block:: python
        :emphasize-lines: 4

        >>> class BarFormatter(BasicFormatter[Any]):
        ...     def print_Bar(self, printer: Printer, item: Bar):
        ...         printer.write("BarFormatter: ")
        ...         self.print(printer, item.bar)
        ...
        ...     def print_str(self, printer: Printer, item: str):
        ...         printer.write(''.join(reversed(item.upper())))
        ...         printer.newline()
        ...
        >>> get_formatter(Bar)(Printer(), Bar("foo"))
        BarFormatter: OOF
        >>> get_formatter(str)(Printer(), "foo")
        StringFormatter: foo

    As you can see, ``self.print(printer, item.bar)`` gives preference to locally defined implementations before doing
    a global lookup for a formatter with a print function.

    We just got "lucky" with that last printout, though, because the ``print_str`` in ``BarFormatter`` has the
    same precedence in the global :func:`get_formatter` lookup as the implementation in ``StringFormatter``. So:

    >>> BarFormatter.DEFAULT_INSTANCE.print(Bar("foo"))
    BarFormatter: OOF
    >>> StringFormatter.DEFAULT_INSTANCE.print("foo")
    StringFormatter: foo

    that will always be true, however, the following might happen:

    >>> get_formatter(str)(Printer(), "foo")
    BarFormatter: OOF

    That behavior might not be desirable. To prevent that (*i.e.*, to compartmentalize the ``BarFormatter``
    implementation of ``print_str`` and *only* use it when expanding a string inside of a ``Bar``),
    :class:`Formtter` classes can be organized hierarchically:

    .. code-block:: python
        :emphasize-lines: 2,8

        >>> class BarStringFormatter(BasicFormatter[str]):
        ...     is_partial = True # This prevents this class from being registered as a global formatter
        ...     def print_str(self, printer: Printer, item: str):
        ...         printer.write(''.join(reversed(item.upper())))
        ...         printer.newline()
        ...
        >>> class BarFormatter(BasicFormatter[Bar]):
        ...     sub_format_types = [BarStringFormatter]
        ...     def print_Bar(self, printer: Printer, item: Bar):
        ...         printer.write("BarFormatter: ")
        ...         self.print(printer, item.bar)
        ...

    Now,

    >>> get_formatter(Bar)(Printer(), Bar("foo"))
    BarFormatter: OOF
    >>> get_formatter(str)(Printer(), "foo")
    StringFormatter: foo

    this will always be the case, and the final command will never invoke the ``BarFormatter`` implementation.

    The sequence of function resolution happens in the ``self.print`` call in ``print_Bar`` follows the
    "Formatting Protocol". It is described in the next section.

    .. _Formatting Protocol:

    Formatting Protocol
    ===================

    The following describes how this module resolves the proper formatter and function to print a given item.

    **Given** an optional :class:`formatter <Formatter>` that is actively being used (*e.g.*, when ``print_Bar`` calls
    ``self.print`` in the ``BarFormatter`` example, above; **and** the ``item`` that is to be formatted.

    * If ``formatter`` is given:
        * For each ``type`` in :meth:`item.__class__.__mro__ <type.__mro__>`:
            * If a print function specifically associated with :obj:`type.__name__` exists in ``formatter``, then use
              that function.
            * Else, repeat this process recursively for any formatters in :attr:`Formatter.sub_format_types`.
            * If none of the subformatters is specialized in :obj:`type`, see if this formatter is the subformatter of
              another parent formatter. If so, repeat this process for the parent.
    * If no formatter has been found by this point, iterate over all other global registered formatters that have not
      yet been tested, and repeat this process given each one.

"""

from abc import ABCMeta, abstractmethod
from functools import partial, wraps
import inspect
import logging
import sys
from typing import Any, Callable, Dict, Generic, Iterable, List, Optional, Sequence, Set, Tuple, Type, TypeVar

if sys.version_info.major == 3 and sys.version_info.minor < 7:
    # Backward compatibility for pre-Python3.7
    from typing import GenericMeta
else:
    # Create a dummy type for GenericMeta since it was removed in Python3.7
    # It was a subclass of ABCMeta in Python3.6, anyway
    GenericMeta = ABCMeta

from .debug import DEBUG_MODE
from .printer import Printer


log = logging.getLogger(__name__)


FORMATTERS: Sequence['Formatter[Any]'] = []
"""A list of default instances of non-partial formatters that have subclassed :class:`Formatter`."""

C = TypeVar("C")


class FormatterError(TypeError):
    pass


class SubFormatterError(FormatterError):
    pass


_PRINTED_WARNINGS: Set[type] = set()


def deprecated_printer(instance_type: type, func: Callable):
    @wraps(func)
    def wrapper(*args, **kwargs):
        if instance_type not in _PRINTED_WARNINGS:
            _PRINTED_WARNINGS.add(instance_type)
            log.warning(f"{instance_type.__name__}.{func.__name__} uses the legacy Graphtage formatting protocol. "
                        f"This may be deprecated in the future. "
                        f"Please update to using the newer `graphtage.Formatter.printer` decorators.")
        return func(*args, **kwargs)

    return wrapper


class FormatterChecker(GenericMeta):
    """The metaclass for :class:`Formatter`.

    For every class that subclasses :class:`Formatter`, if :attr:`Formatter.is_partial` is :const:`False` (the default)
    and if the class is not abstract, then an instance of that class is automatically constructed and added to the
    global list of formatters. This same automatically constructed instance will be assigned to the
    :attr:`Formatter.DEFAULT_INSTANCE` attribute.

    All methods of the subclass that begin with "``print_``" will be verified insofar as it is possible.

    """
    def __init__(cls, name, bases, clsdict):
        """Initializes the formatter checker.

        Raises:
            TypeError: If :obj:`cls` defines a method starting with "``print``" that has a keyword argument ``printer``
                with a type hint that is not a subclass of :class:`graphtage.printer.Printer`.

        """

        if len(cls.mro()) > 2 and not cls.__abstractmethods__:
            # Instantiate a version of the Formatter to add it to our global dicts:
            # If the formatter has a custom init, we can't use it as a default
            try:
                if hasattr(cls, '__init__') and inspect.signature(cls.__init__).bind(None):
                    pass
                # if we got this far then the Formatter class has a default constructor
                instance = cls()
            except SubFormatterError:
                raise
            except TypeError as e:
                log.debug(f"Formatter {name} cannot be instantiated as a default constructor: {e!s}")
                instance = None
            if instance is not None:
                if not hasattr(instance, "PRINTERS") or instance.PRINTERS is None:
                    setattr(instance, "PRINTERS", {})
                else:
                    setattr(instance, "PRINTERS", dict(instance.PRINTERS))
                print_funcs = []
                for member, member_type in inspect.getmembers(instance):
                    if member.startswith('print_'):
                        sig = inspect.signature(member_type)
                        if 'printer' in sig.parameters:
                            a = sig.parameters['printer'].annotation
                            if a is not None and a != inspect.Signature.empty and inspect.isclass(a) \
                                    and not issubclass(Printer, a):
                                raise TypeError(f"The type annotation for {name}.{member}(printer: {a}) was expected "
                                                f"to be a superclass of graphtage.printer.Printer")
                        print_funcs.append(member)
                for member in print_funcs:
                    setattr(instance, member, deprecated_printer(instance_type=cls, func=getattr(instance, member)))
                if not instance.is_partial and not cls.__name__ == 'BasicFormatter':
                    FORMATTERS.append(instance)
                setattr(cls, '_DEFAULT_INSTANCE', instance)
        super().__init__(name, bases, clsdict)

    @property
    def DEFAULT_INSTANCE(cls: Type[C]) -> C:
        """A default instance of this formatter, automatically instantiated.

        If this formatter is abstract, retrieving this property will raise a FormatterError.

        """
        if not hasattr(cls, "_DEFAULT_INSTANCE") or cls._DEFAULT_INSTANCE is None:
            if cls.__abstractmethods__:
                abstract = [f"`{m}`" for m in sorted(cls.__abstractmethods__)]
                raise FormatterError(f"{cls.__name__} does not have a DEFAULT_INSTANCE because it cannot be "
                                     f"instantiated; abstract method{['', 's'][len(abstract) > 1]} "
                                     f"{', '.join(abstract)} must be implemented")
            raise FormatterError(f"{cls.__name__} does not have a DEFAULT_INSTANCE because an unknown error occurred "
                                 f"during its instantiation")

        return cls._DEFAULT_INSTANCE


T = TypeVar('T')


def _get_formatter(
        node_type: Type[T],
        base_formatter: 'Formatter',
        tested: Set[Type['Formatter']],
        test_parent: bool = True
) -> Optional[Tuple["Formatter", Type[T], Callable[[Printer, T], Any]]]:
    ret = base_formatter.resolve_printer(node_type, cls_instance=base_formatter, ignore_formatters=tested)
    if ret is not None:
        return ret
    formatter_stack = [base_formatter]
    while formatter_stack:
        formatter = formatter_stack.pop()
        if formatter.__class__ not in tested:
            grandchildren = []
            for c in node_type.mro():
                if hasattr(formatter, f'print_{c.__name__}'):
                    return formatter, c, getattr(formatter, f'print_{c.__name__}')
                for sub_formatter in formatter.sub_formatters:
                    if sub_formatter not in tested:
                        if hasattr(sub_formatter, f'print_{c.__name__}'):
                            return formatter, c, getattr(sub_formatter, f'print_{c.__name__}')
                        grandchildren.extend(sub_formatter.sub_formatters)
            tested.add(formatter.__class__)
            tested |= set(s.__class__ for s in formatter.sub_formatters)
            formatter_stack.extend(reversed(grandchildren))
    if test_parent:
        parent = base_formatter.parent
        while parent is not None:
            if parent.__class__ not in tested:
                new_instance = parent.__class__()
                new_instance.parent = base_formatter
                ret = _get_formatter(node_type, new_instance, tested, test_parent=False)
                if ret is not None:
                    return ret
            parent = parent.parent
    return None


def get_formatter(
        node_type: Type[T],
        base_formatter: Optional['Formatter'] = None
) -> Optional[Callable[[Printer, T], Any]]:
    """Uses the :ref:`Formatting Protocol` to determine the correct formatter for a given type.

    See :ref:`this section <What the formatter module can do>` for a number of examples.

    Args:
        node_type: The type of the object to be formatted.
        base_formatter: An existing formatter from which the request is being made. This will affect the formatter
            resolution according to the :ref:`Formatting Protocol`.

    Returns: The formatter for object type :obj:`node_type`, or :const:`None` if none was found.

    """
    tested_formatters: Set[Type[Formatter]] = set()
    if base_formatter is not None:
        matched = _get_formatter(node_type, base_formatter, tested_formatters)
        if matched is not None:
            _, _, ret = matched
            return ret

    lowest_mro_depth = len(node_type.mro()) + 1
    best_possibility: Optional[Callable[[Printer, T], Any]] = None
    for formatter in FORMATTERS:
        if formatter.__class__ not in tested_formatters:
            if base_formatter is not None:
                formatter = formatter.__class__()
                formatter.parent = base_formatter
            matched = _get_formatter(node_type, formatter, tested_formatters)
            if matched is not None:
                _, matched_type, ret = matched
                mro_depth = len(matched_type.mro())
                if mro_depth < lowest_mro_depth:
                    lowest_mro_depth = mro_depth
                    best_possibility = ret
    return best_possibility


class Formatter(Generic[T], metaclass=FormatterChecker):
    """"""

    sub_format_types: Sequence[Type['Formatter[T]']] = ()
    """A list of formatter types that should be used as sub-formatters in the :ref:`Formatting Protocol`."""
    sub_formatters: List['Formatter[T]'] = []
    """The list of instantiated formatters corresponding to :attr:`Formatter.sub_format_types`.
    
    This list is automatically populated by :meth:`Formatter.__new__` and should never be manually modified.
    
    """
    parent: Optional['Formatter[T]'] = None
    """The parent formatter for this formatter instance.
    
    This is automatically populated by :meth:`Formatter.__new__` and should never be manually modified.

    """
    is_partial: bool = False
    """A flag indicating whether or not this formatter is partial.
    
    If `True`, this formatter will be considered on every print that is not resolvable from the base formatter.
     
    """
    PRINTERS: Dict[Type[T], Callable[["Formatter", Printer, T], Any]]

    @property
    def root(self) -> 'Formatter[T]':
        """Returns the root formatter."""
        node = self
        while node.parent is not None:
            node = node.parent
        return node

    @staticmethod
    def printer(item_type: Type[T]):
        def wrapper(func: Callable[[C, Printer, T], Any]) -> Callable[[C, Printer, T], Any]:
            if hasattr(func, "_printer_for_type"):
                func._printer_for_type = func._printer_for_type + (item_type,)
            else:
                setattr(func, "_printer_for_type", (item_type,))
            return func

        return wrapper

    def __new__(cls, *args, **kwargs) -> 'Formatter[T]':
        """Instantiates a new formatter.

        This automatically instantiates and populates :attr:`Formatter.sub_formatters` and sets their
        :attr:`parent<Formatter.parent>` to this new formatter.

        """
        if cls.__abstractmethods__:
            args_str = [str(a) for a in args]
            kwarg_str = [f"{k!s}={v!r}" for k, v in kwargs.items()]
            raise FormatterError(f"Cannot instantiate {cls.__name__}({', '.join(args_str + kwarg_str)}) because "
                                 f"it has unimplemented abstract methods: {cls.__abstractmethods__!r}")
        ret: Formatter[T] = super().__new__(cls, *args, **kwargs)
        setattr(ret, 'sub_formatters', [])
        for sub_formatter in ret.sub_format_types:
            try:
                ret.sub_formatters.append(sub_formatter())
                ret.sub_formatters[-1].parent = ret
            except TypeError as e:
                raise SubFormatterError(f"Error instantiating {cls.__name__}'s sub-formatter of type "
                                        f"{sub_formatter.__name__}; its default constructor raised: {e!s}. "
                                        f"Sub-formatters must be instantiable via a constructor with no arguments.")
        return ret

    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)
        if not hasattr(cls, "PRINTERS") or cls.PRINTERS is None:
            setattr(cls, "PRINTERS", {})
        else:
            setattr(cls, "PRINTERS", dict(cls.PRINTERS))
        new_printers: Dict[Type[T], Callable[["Formatter", Printer, T], Any]] = {}
        for member_name, member in cls.__dict__.items():
            if member_name.startswith("__"):
                continue
            if hasattr(member, "_printer_for_type"):
                for printer_type in getattr(member, "_printer_for_type"):
                    if not isinstance(printer_type, type):
                        raise TypeError(f"{cls.__name__}.{member_name} was registered as a printer for "
                                        f"{printer_type!r}, which is not a type")
                    elif printer_type in cls.PRINTERS and cls.PRINTERS[printer_type].__name__ != member_name:
                        raise TypeError(f"A printer for type {printer_type.__name__} is already registered to "
                                        f"{cls.PRINTERS[printer_type]!r} and cannot be re-registered to "
                                        f"{cls.__name__}.{member_name}")
                    elif printer_type in new_printers:
                        raise TypeError(f"A printer for type {printer_type.__name__} is already registered to "
                                        f"{new_printers[printer_type]!r} and cannot be re-registered to "
                                        f"{cls.__name__}.{member_name}")
                    new_printers[printer_type] = member  # type: ignore
        cls.PRINTERS.update(new_printers)

    @classmethod
    def resolve_printer(
            cls: Type[C], item_type: Type[T], cls_instance: Optional[C] = None,
            ignore_formatters: Iterable[Type["Formatter"]] = ()
    ) -> Optional[Tuple["Formatter", Type[T], Callable[[Printer, T], Any]]]:
        if cls_instance is None:
            cls_instance = cls.DEFAULT_INSTANCE
        ignore_formatters = frozenset(ignore_formatters)
        lowest_mro_depth = len(item_type.mro()) + 1
        best_possibility: Optional[Tuple["Formatter", Type[T], Callable[[Printer, T], Any]]] = None
        for sub_type in cls_instance.sub_format_types:
            if sub_type in ignore_formatters:
                continue
            sub = sub_type()
            sub.parent = cls_instance
            sub_printer = sub_type.resolve_printer(item_type, cls_instance=sub, ignore_formatters=ignore_formatters)
            if sub_printer is not None:
                _, matched_type, _ = sub_printer
                for i, t in enumerate(item_type.mro()):
                    if t == matched_type:
                        if best_possibility is None or i < lowest_mro_depth:
                            lowest_mro_depth = i
                            best_possibility = sub_printer
                        break
                else:
                    raise NotImplementedError("This should never happen")
        if cls not in ignore_formatters:
            for i, t in enumerate(item_type.mro()):
                if best_possibility is not None and i >= lowest_mro_depth:
                    break
                if t in cls.PRINTERS:
                    return cls_instance, t, partial(cls.PRINTERS[t], cls_instance)
        return best_possibility

    @classmethod
    def get_printer(
            cls: Type[C], item_type: Type[T], cls_instance: Optional[C] = None
    ) -> Optional[Callable[[Printer, T], Any]]:
        match = cls.resolve_printer(item_type, cls_instance=cls_instance)
        if match is None:
            return None
        _, _, ret = match
        return ret

    def get_formatter(self, item: T) -> Optional[Callable[[Printer, T], Any]]:
        """Looks up a formatter for the given item using this formatter as a base.

        Equivalent to::

            get_formatter(item.__class__, base_formatter=self)

        """
        return get_formatter(item.__class__, base_formatter=self)

    @abstractmethod
    def print(self, printer: Printer, item: T):
        """Prints an item to the printer using the proper formatter.

        This method is abstract because subclasses should decide how to handle the case when a formatter was not found
        (*i.e.*, when :meth:`self.get_formatter<Formatter.get_formatter>` returns :const:`None`).

        """
        raise NotImplementedError()


if sys.version_info.major == 3 and sys.version_info.minor < 7:
    # Backward compatibility for pre-Python3.7
    basic_formatter_types = (Formatter,)
else:
    basic_formatter_types = (Generic[T], Formatter[T])


class BasicFormatter(*basic_formatter_types):
    """A basic formatter that falls back on an item's natural string representation if no formatter is found."""

    def print(self, printer: Printer, item: T):
        """Prints the item to the printer.

        This is equivalent to::

            formatter = self.get_formatter(item)
            if formatter is None:
                return printer.write(str(item))
            else:
                return formatter(printer, item)

        """
        formatter = self.get_formatter(item)
        if formatter is None:
            return printer.write(str(item))
        else:
            return formatter(printer, item)
