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

.. _consuming-diffs:

Inspecting and Consuming Diffs
------------------------------

Use :func:`graphtage.pydiff.diff` to keep the result of a comparison, or call
:meth:`graphtage.TreeNode.diff` on trees built by a filetype. The result is an annotated copy of the **source**
tree, not the destination tree or a patch applied to the source. Existing children, including removed children,
remain in :meth:`graphtage.TreeNode.children`; insertions are recorded separately on their container::

    >>> from graphtage import EditedTreeNode, Match
    >>> from graphtage.pydiff import diff
    >>> before = {"a": 1, "b": [1, 3], "c": [4, 5]}
    >>> after = {"a": 2, "b": [1, 2, 3], "c": [5]}
    >>> result = diff(before, after)
    >>> isinstance(result, EditedTreeNode)
    True
    >>> result.to_obj() == before
    True
    >>> items = next(entry.value for entry in result.children() if entry.key.to_obj() == "b")
    >>> [child.to_obj() for child in items.children()], [node.to_obj() for node in items.inserted]
    ([1, 3], [2])
    >>> isinstance(items.inserted[0], EditedTreeNode)
    False

Each node in the source copy also retains its original node type and has these
:class:`graphtage.EditedTreeNode` attributes:

.. list-table:: Diff annotations
   :header-rows: 1
   :widths: 20 80

   * - Attribute
     - Meaning
   * - ``removed``
     - A :class:`graphtage.Remove` marked this node for removal. Descendants of a removed subtree need not
       each be marked.
   * - ``inserted``
     - Nodes added to this container by :class:`graphtage.Insert`. They come from the destination tree and
       need not be :class:`graphtage.EditedTreeNode` instances. There is no ``inserted`` flag on a new child.
   * - ``matched_to``
     - The destination node assigned by :class:`graphtage.Match`, including zero-cost matches.
       Other edit types need not set it.
   * - ``edit``
     - The edit assigned to this node, or ``None``. For example, matching an unchanged container can leave
       its descendants' ``edit`` attributes unset.
   * - ``edit_list``
     - Edits assigned to this node by the diff callbacks. Container edits can be compound, so concatenating
       these lists over the tree is not a flat list of changes and can count the same change more than once.

A :class:`graphtage.Match` does not necessarily mean "unchanged". Numeric value changes normally use a non-zero-cost
``Match``, so checking only for :class:`graphtage.Replace` misses them. Check
:meth:`graphtage.Edit.has_non_zero_cost` after guarding against ``edit is None``::

    >>> scalar = diff(1, 2)
    >>> isinstance(scalar.edit, Match), scalar.edit.has_non_zero_cost()
    (True, True)
    >>> scalar.to_obj(), scalar.matched_to.to_obj()
    (1, 2)

Strings can use :class:`graphtage.StringEdit`, which does not populate ``matched_to``. For changes other than
insertions and removals, ``edit.to_node`` gives the destination node. A ``Replace`` represents a wholesale
replacement, which can involve a container.

The following iterator reports removed and inserted subtrees, changed leaves (including dictionary keys), and
wholesale replacements. It visits the root too, checks insertions on every container, and stops at removed or
replaced subtrees. It does not report compound container edits again when reporting their descendants::

    >>> from graphtage import LeafNode, Replace
    >>> def summarize(node):
    ...     if node.removed:
    ...         yield ("removed", node.to_obj())
    ...         return
    ...     edit = node.edit
    ...     if edit is not None and edit.has_non_zero_cost():
    ...         if isinstance(node, LeafNode) or isinstance(edit, Replace):
    ...             yield ("changed", node.to_obj(), edit.to_node.to_obj())
    ...             return
    ...     for inserted in node.inserted:
    ...         yield ("inserted", inserted.to_obj())
    ...     for child in node.children():
    ...         yield from summarize(child)
    >>> list(summarize(result))
    [('changed', 1, 2), ('inserted', 2), ('removed', 4)]
    >>> list(summarize(diff([], [2])))
    [('inserted', 2)]
    >>> list(summarize(diff("cat", "cut")))
    [('changed', 'cat', 'cut')]
    >>> list(summarize(diff([[4, 5]], [])))
    [('removed', [4, 5])]
    >>> list(summarize(diff([1], 42)))
    [('changed', [1], 42)]
    >>> list(summarize(diff([1], [1])))
    []

:meth:`graphtage.TreeNode.to_obj` reads a node's stored value; it does not apply diff annotations. For scalar,
list, and mapping nodes, it returns Python values suitable for downstream processing. Convert an inserted node
or an edit's destination node to obtain its new value. Representations are node-specific: for example,
``KeyValuePairNode.to_obj()`` returns a pair of nodes, and custom objects need not round-trip to their original
class. Use :meth:`graphtage.TreeNode.copy` and :meth:`graphtage.TreeNode.copy_from` when building your own tree
representation of changes, with copies of any children that already have parents.

Locating Edits with Contexts
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

When you need to know where a change happened, start with
:meth:`graphtage.TreeNode.get_all_edit_contexts` on the **two input trees**. This computes edits without annotating
the source and yields ``(ancestor_path, edit)`` pairs. It expands compound edits and omits zero-cost matches.
:meth:`graphtage.TreeNode.get_all_edits` yields the same edits without paths, and the command-line
``--edit-digest`` option uses the context API.

In the current implementation, the path starts with the destination root; subsequent entries follow
``from_node`` through the nested sub-edits. A root-level scalar edit therefore has only the destination root in
its path. These are edit-context paths, not guaranteed parent/child chains in a single tree or ready-made JSON
Pointers. For an insertion the final entry is the inserted destination node, while ``edit.insert_into`` names
the source container. Use the edit's attributes to distinguish the operation::

    >>> from graphtage import Insert, KeyValuePairNode, Remove
    >>> from graphtage.pydiff import build_tree
    >>> def describe_edit(edit):
    ...     if isinstance(edit, Insert):
    ...         return ("inserted", edit.to_insert.to_obj())
    ...     if isinstance(edit, Remove):
    ...         return ("removed", edit.from_node.to_obj())
    ...     return ("changed", edit.from_node.to_obj(), edit.to_node.to_obj())
    >>> source, target = build_tree(before), build_tree(after)
    >>> contexts = list(source.get_all_edit_contexts(target))
    >>> records = []
    >>> for ancestors, edit in contexts:
    ...     keys = tuple(n.key.to_obj() for n in ancestors[1:] if isinstance(n, KeyValuePairNode))
    ...     records.append((keys, describe_edit(edit)))
    >>> records
    [(('a',), ('changed', 1, 2)), (('b',), ('inserted', 2)), (('c',), ('removed', 4))]
    >>> ancestors, insertion = contexts[1]
    >>> ancestors[0] is target, ancestors[-1] is insertion.to_insert
    (True, True)
    >>> insertion.insert_into.to_obj(), insertion.to_insert.to_obj()
    ([1, 3], 2)
    >>> source.edited, target.edited
    (False, False)
    >>> scalar_source, scalar_target = build_tree(1), build_tree(2)
    >>> [(path == (scalar_target,), describe_edit(edit))
    ...  for path, edit in scalar_source.get_all_edit_contexts(scalar_target)]
    [(True, ('changed', 1, 2))]
    >>> list(build_tree([1]).get_all_edit_contexts(build_tree([1])))
    []

Here ``keys`` identifies enclosing dictionary entries only; it is not a complete location for list elements or
key edits. :meth:`graphtage.TreeNode.print_parent_context` can render a node's location relative to its parent,
as ``--edit-digest`` does. For machine-readable locations, interpret the container types for your input model.
In particular, ``inserted`` alone does not encode insertion positions; consult the container's edit sequence
when order matters. ``Insert.from_node`` is the node being inserted and ``Insert.to_node`` is the container;
``Remove.to_node`` is also the container, not a replacement value.

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
