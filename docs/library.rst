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

Transforming Nodes with Edits
-----------------------------

To see the sequence of edits to transform this tree to another, we call :meth:`graphtage.TreeNode.get_all_edits`::

    >>> to_tree = json.build_tree({"bar": [2, 3, 4]})
    >>> to_tree
    DictNode([KeyValuePairNode(key=StringNode('bar'), value=ListNode((IntegerNode(2), IntegerNode(3), IntegerNode(4))))])
    >>> for edit in from_tree.get_all_edits(to_tree):
    ...     print(edit)
    Remove(IntegerNode(1), remove_from=ListNode((IntegerNode(1), IntegerNode(2), IntegerNode(3), IntegerNode(4))))
    StringEdit(from_node=StringNode('foo'), to_node=StringNode('bar'))

Applying Edits to Nodes
-----------------------

Both nodes and edits are immutable. We can perform a diff to apply edits to nodes, producing a new tree constructed of
:class:`graphtage.EditedTreeNode` objects. Using some Python magic, the new tree's nodes maintain all of the same
characteristics of the source nodes—including their source node class types—but are *also* :func:`instanceof`
:class:`graphtage.EditedTreeNode`, too.

Here is how to diff two nodes::

    >>> from_node.diff(to_node)
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
to a stream. For example, to print our diff in JSON format to the default printer (STDOUT), we would do::

    >>> from graphtage import printer
    >>> with printer.DEFAULT_PRINTER as p:
    ...     json.JSONFormatter.DEFAULT_INSTANCE.print(printer.DEFAULT_PRINTER, diff)
    ...
    {
        "++bar++~~foo~~": [
            ~~1~~,
            2,
            3,
            4
        ]
    }

Since Graphtage's formatters are independent of the input format, thanks to the intermediate representation, we can
just as easily output the diff in another format, like YAML::

    >>> from graphtage import yaml
    >>> with printer.DEFAULT_PRINTER as p:
    ...     yaml.YAMLFormatter.DEFAULT_INSTANCE.print(printer.DEFAULT_PRINTER, diff)
    ...
    ++bar++~~foo~~:
    - ~~1~~
    - 2
    - 3
    - 4

Diffing In-Memory Python Objects
--------------------------------

When used as a library, Graphtage has the ability to diff in-memory Python objects. This can be useful when debugging,
for example, to quickly determine the difference between two Python objects that cause a differential.::

    >>> from graphtage.pydiff import print_diff
    >>> with printer.DEFAULT_PRINTER as p:
    ...     obj1 = [1, 2, {3: "three"}, 4]
    ...     obj2 = [1, 2, {3: 3}, "four"]
    ...     print_diff(obj1, obj2, printer=p)
    [1,2,{3: "three" -> 3},++"four"++~~4~~]

Python object diffing also works with custom classes::

    >>> class Foo:
    ...     def __init__(self, bar, baz):
    ...         self.bar = bar
    ...         self.baz = baz
    >>> with printer.DEFAULT_PRINTER as p:
    ...     print_diff(Foo("bar", "baz"), Foo("bar", "bak"), printer=p)
    Foo(bar="bar", baz="ba++k++~~z~~")
