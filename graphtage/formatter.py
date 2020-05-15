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

import inspect
import logging
import sys
from abc import ABCMeta, abstractmethod
from typing import Any, Callable, Generic, List, Optional, Sequence, Set, Type, TypeVar

if sys.version_info.major == 3 and sys.version_info.minor < 7:
    # Backward compatibility for pre-Python3.7
    from typing import GenericMeta
else:
    # Create a dummy type for GenericMeta since it was removed in Python3.7
    # It was a subclass of ABCMeta in Python3.6, anyway
    GenericMeta = ABCMeta

from .printer import Printer


log = logging.getLogger(__name__)


FORMATTERS: Sequence['Formatter[Any]'] = []
"""A list of default instances of non-partial formatters that have subclassed :class:`Formatter`."""


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
                if not instance.is_partial and not cls.__name__ == 'BasicFormatter':
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
    """"""

    DEFAULT_INSTANCE: 'Formatter[T]' = None
    """A default instance of this formatter, automatically instantiated by the :class:`FormatterChecker` metaclass."""
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

    @property
    def root(self) -> 'Formatter[T]':
        """Returns the root formatter."""
        node = self
        while node.parent is not None:
            node = node.parent
        return node

    def __new__(cls, *args, **kwargs) -> 'Formatter[T]':
        """Instantiates a new formatter.

        This automatically instantiates and populates :attr:`Formatter.sub_formatters` and sets their
        :attr:`parent<Formatter.parent>` to this new formatter.

        """
        ret: Formatter[T] = super().__new__(cls, *args, **kwargs)
        setattr(ret, 'sub_formatters', [])
        for sub_formatter in ret.sub_format_types:
            ret.sub_formatters.append(sub_formatter())
            ret.sub_formatters[-1].parent = ret
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
                printer.write(str(item))
            else:
                formatter(printer, item)

        """
        f = self.get_formatter(item)
        if f is None:
            printer.write(str(item))
        else:
            f(printer, item)
