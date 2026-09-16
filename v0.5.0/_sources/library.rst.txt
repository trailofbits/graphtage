Using Graphtage Programmatically
================================

Graphtage is a command line utility, but it can just as easily be used as a library. This section documents how to
interact with Graphtage directly from Python.

The Intermediate Representation
-------------------------------

Graphtage's diffing algorithms operate on an
`intermediate representation <https://en.wikipedia.org/wiki/Intermediate_representation>`__ rather than on the data
structures of the original file format. This allows Graphtage to have generic comparison algorithms that can work on
*any* input file type. The intermediate representation is a tree of :class:`graphtage.TreeNode` objects.

Therefore, the first step is to convert the files being diffed into Graphtage's intermediate representation. The JSON
filetype has a function to convert arbitrary Python objects (comprised of standard Python types) into Graphtage trees::

    >>> from graphtage import json
    >>> from_tree = json.build_tree({"foo": [1, 2, 3, 4]})
    >>> from_tree
    DictNode([KeyValuePairNode(key=StringNode('foo'), value=ListNode((IntegerNode(1), IntegerNode(2), IntegerNode(3), IntegerNode(4))))])

Build Options
-------------

Every ``build_tree`` function—:func:`graphtage.json.build_tree`, :func:`graphtage.pydiff.build_tree`, and the
:meth:`build_tree <graphtage.Filetype.build_tree>` method of every :class:`graphtage.Filetype`—accepts an optional
:obj:`options` argument of type :class:`graphtage.BuildOptions`. It controls which edits the matching algorithms are
allowed to consider, which is the main lever you have over both the shape of the diff and how long it takes to
compute::

    >>> from graphtage import BuildOptions, json
    >>> options = BuildOptions(ignore_list_order=True)
    >>> json.build_tree([1, 2, 3], options=options)
    UnorderedListNode([IntegerNode(1), IntegerNode(2), IntegerNode(3)])

Passing no options is the same as passing ``BuildOptions()``. The options are:

:attr:`allow_key_edits <graphtage.BuildOptions.allow_key_edits>`
    Whether to consider editing keys when matching :class:`graphtage.KeyValuePairNode` objects. Defaults to
    :const:`True`. With :const:`False`, dictionary entries only match when their keys are equal, and the tree is built
    from :class:`graphtage.FixedKeyDictNode` rather than :class:`graphtage.DictNode`. This is the least expensive
    dictionary strategy.

:attr:`auto_match_keys <graphtage.BuildOptions.auto_match_keys>`
    Whether to automatically match key/value pairs in dictionaries if they share the same key. Defaults to
    :const:`True`. Key edits are still considered for the pairs that are left over. Setting this to :const:`False`
    while leaving :attr:`allow_key_edits <graphtage.BuildOptions.allow_key_edits>` set costs every possible pairing of
    keys, which is the most expensive dictionary strategy.

:attr:`allow_list_edits <graphtage.BuildOptions.allow_list_edits>`
    Whether to consider insert and remove edits to lists. Defaults to :const:`True`. With :const:`False`, lists are
    compared element by element.

:attr:`ignore_list_order <graphtage.BuildOptions.ignore_list_order>`
    Whether to match the elements of a list as an unordered collection. Defaults to :const:`False`. See
    :ref:`Unordered Lists` for the trade-off this makes.

:attr:`check_for_cycles <graphtage.BuildOptions.check_for_cycles>`
    Whether to check the input for cycles while building the tree. Defaults to :const:`True`. In-memory Python objects
    can refer to themselves, and walking such an object without this check does not terminate.

:attr:`ignore_cycles <graphtage.BuildOptions.ignore_cycles>`
    What to do when a cycle is found. Defaults to :const:`False`, which raises a :exc:`ValueError`. With :const:`True`,
    the back edge is replaced by a :class:`graphtage.builder.CyclicReference` leaf and the walk continues::

        >>> from graphtage import BuildOptions
        >>> from graphtage.pydiff import build_tree
        >>> cyclic = []
        >>> cyclic.append(cyclic)
        >>> build_tree(cyclic)
        Traceback (most recent call last):
          ...
        ValueError: Detected a cycle in [[...]] at child [[...]]
        >>> build_tree(cyclic, options=BuildOptions(ignore_cycles=True))
        ListNode((CyclicReference(<graphtage.object_set.IdentityHash object at 0x10df992b0>),))

:attr:`printer <graphtage.BuildOptions.printer>`
    The printer used to report progress while building the tree. Defaults to
    :attr:`graphtage.printer.NULL_PRINTER`, which prints nothing. The command line client sets this to the printer it
    is using, which is why ``graphtage`` shows a progress bar and the library does not.

Any keyword that :class:`graphtage.BuildOptions` does not recognize is set as an attribute, and any attribute that was
never set reads back as :const:`False`. That makes it possible for a :class:`graphtage.Filetype` to define its own
options, but it also means that a misspelled option name is silently accepted.

Transforming Nodes with Edits
-----------------------------

To see the sequence of edits to transform this tree to another, we call :meth:`graphtage.TreeNode.get_all_edits`::

    >>> to_tree = json.build_tree({"bar": [2, 3, 4]})
    >>> to_tree
    DictNode([KeyValuePairNode(key=StringNode('bar'), value=ListNode((IntegerNode(2), IntegerNode(3), IntegerNode(4))))])
    >>> for edit in from_tree.get_all_edits(to_tree):
    ...     print(edit)
    StringEdit(from_node=StringNode('foo'), to_node=StringNode('bar'))
    Remove(IntegerNode(1), remove_from=ListNode((IntegerNode(1), IntegerNode(2), IntegerNode(3), IntegerNode(4))))

Applying Edits to Nodes
-----------------------

Both nodes and edits are immutable. We can perform a diff to apply edits to nodes, producing a new tree constructed of
:class:`graphtage.EditedTreeNode` objects. Using some Python magic, the new tree's nodes maintain all of the same
characteristics of the source nodes—including their source node class types—but are *also* :func:`isinstance`
:class:`graphtage.EditedTreeNode`, too.

Here is how to diff two nodes::

    >>> diff = from_tree.diff(to_tree)
    >>> diff
    EditedDictNode([EditedKeyValuePairNode(key=EditedStringNode('foo'), value=EditedListNode((EditedIntegerNode(1), EditedIntegerNode(2), EditedIntegerNode(3), EditedIntegerNode(4))))])

As you can see, the tree was reconstructed with edited versions of each node. Each node will have a new member variable,
:attr:`graphtage.EditedTreeNode.edit`, containing the edit that that chose to apply to itself (or :const:`None` if the
node did not need to be edited). There are also additional member variables to indicate whether the node has been
removed from its parent container.

Formatting and Printing Results
-------------------------------

There are two components to outputting a tree or diff: a :class:`graphtage.formatter.Formatter`, which is responsible
for the syntax of the output, and a :class:`graphtage.printer.Printer`, which is responsible for rendering that output
to a stream. Read the default printer with :func:`graphtage.printer.get_default_printer` rather than importing
:attr:`graphtage.printer.DEFAULT_PRINTER` by name, because :func:`graphtage.printer.set_default_printer` rebinds the
module attribute and an imported name never sees the replacement. For example, to print our diff in JSON format to the
default printer (STDOUT), we would do::

    >>> from graphtage import printer
    >>> with printer.get_default_printer() as p:
    ...     json.JSONFormatter.DEFAULT_INSTANCE.print(p, diff)
    ...
    {
        "~~foo~~++bar++": [
            ~~1~~,
            2,
            3,
            4
        ]
    }

Since Graphtage's formatters are independent of the input format, thanks to the intermediate representation, we can
just as easily output the diff in another format, like YAML::

    >>> from graphtage import yaml
    >>> with printer.get_default_printer() as p:
    ...     yaml.YAMLFormatter.DEFAULT_INSTANCE.print(p, diff)
    ...
    ~~foo~~++bar++: 
    - ~~1~~
    - 2
    - 3
    - 4

Diffing In-Memory Python Objects
--------------------------------

When used as a library, Graphtage has the ability to diff in-memory Python objects. This can be useful when debugging,
for example, to quickly determine the difference between two Python objects that cause a differential.::

    >>> from graphtage.pydiff import print_diff
    >>> with printer.get_default_printer() as p:
    ...     obj1 = [1, 2, {3: "three"}, 4]
    ...     obj2 = [1, 2, {3: 3}, "four"]
    ...     print_diff(obj1, obj2, printer=p)
    [1,2,{3: "three" -> 3},4 -> "four"]

Python object diffing also works with custom classes::

    >>> class Foo:
    ...     def __init__(self, bar, baz):
    ...         self.bar = bar
    ...         self.baz = baz
    >>> with printer.get_default_printer() as p:
    ...     print_diff(Foo("bar", "baz"), Foo("bar", "bak"), printer=p)
    Foo(bar="bar", baz="ba~~z~~++k++")

:func:`graphtage.pydiff.print_diff` only prints. To inspect a diff instead, call :func:`graphtage.pydiff.diff`, which
returns the same edited tree that :meth:`graphtage.TreeNode.diff` produces::

    >>> from graphtage.pydiff import diff
    >>> d = diff(obj1, obj2)
    >>> d
    EditedListNode((EditedIntegerNode(1), EditedIntegerNode(2), EditedDictNode([EditedKeyValuePairNode(key=EditedIntegerNode(3), value=EditedStringNode('three'))]), EditedIntegerNode(4)))
    >>> d.edit.bounds()
    Range(9, 9)
    >>> [child.edit for child in d.children() if child.edit.has_non_zero_cost()]
    [<graphtage.multiset.MultiSetEdit object at 0x10d72b4a0>, Match(match_from=EditedIntegerNode(4), match_to=StringNode('four'), cost=4)]

:func:`graphtage.pydiff.build_tree` is the underlying tree builder. It accepts any Python object, including instances
of classes that Graphtage has never seen, and represents them as a :class:`graphtage.pydiff.PyObj` node whose children
are the object's public attributes::

    >>> from graphtage.pydiff import build_tree
    >>> build_tree(Foo("bar", "baz"))
    PyObj(class_name=StringNode('Foo'), attrs=PyObjAttributes([KeywordArgument(key=StringNode('bar'), value=StringNode('bar')), KeywordArgument(key=StringNode('baz'), value=StringNode('baz'))]))

Both functions accept the same :obj:`options` argument as the filetype tree builders.

Diffing Python Source Code
--------------------------

Graphtage can also diff Python source code by way of the standard library's :mod:`ast` module.
:func:`graphtage.pydiff.ast_to_tree` converts an :class:`ast.AST` into an intermediate representation built from the
node types in :mod:`graphtage.ast`: :class:`graphtage.ast.Module` for a source file,
:class:`graphtage.ast.Assignment`, :class:`graphtage.ast.Call`, :class:`graphtage.ast.Import`,
:class:`graphtage.ast.Subscript`, and :class:`graphtage.ast.KeywordArgument`. The conversion is performed by
:class:`graphtage.pydiff.ASTBuilder`, a :class:`graphtage.builder.BasicBuilder` subclass that covers a subset of
Python's syntax; a syntax node that it has no builder for raises :exc:`NotImplementedError`.

:class:`graphtage.pydiff.PyDiffFormatter` renders the result back as Python source::

    >>> import ast
    >>> from graphtage import printer
    >>> from graphtage.pydiff import PyDiffFormatter, ast_to_tree
    >>> from_ast = ast_to_tree(ast.parse("from foo import bar\nx = bar(1, 2)\n"))
    >>> from_ast
    Module((Import(names=ListNode((PyAlias(name=StringNode('bar'), as_name=StringNode('')),)), from_name=StringNode('foo')), Assignment(targets=ListNode((StringNode('x'),)), value=Call(func=StringNode('bar'), args=CallArguments((IntegerNode(1), IntegerNode(2))), kwargs=CallKeywords([])))))
    >>> to_ast = ast_to_tree(ast.parse("from foo import bar\nx = bar(1, 2)\ny = bar(3)\n"))
    >>> with printer.get_default_printer() as p:
    ...     PyDiffFormatter.DEFAULT_INSTANCE.print(p, from_ast.diff(to_ast))
    ...
    from foo import bar
    x = bar(1, 2)
    ++y = bar(3)++

Like the other tree builders, :func:`graphtage.pydiff.ast_to_tree` accepts an :obj:`options` argument. Its list and
tuple builder deliberately ignores :attr:`graphtage.BuildOptions.ignore_list_order`, because the elements of a Python
list or tuple literal are positional.
