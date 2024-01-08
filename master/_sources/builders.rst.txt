.. _Builders:

Constructing Graphtage Trees
============================

Graphtage operates on trees represented by the :class:`graphtage.TreeNode` base class.
There are various predefined specializations of tree nodes, such as :class:`graphtage.IntegerNode` for integers, :class:`graphtage.ListNode` for lists, and :class:`graphtage.DictNode` for dictionaries. :class:`graphtage.TreeNode` has an optional :attr:`parent <graphtage.TreeNode.parent>` and a potentially empty set of :func:`children <graphtage.TreeNode.children>`.

Graphtage provides a :class:`graphtage.builder.Builder` class for conveniently converting arbitrary objects into a tree of :class:`TreeNode <graphtage.TreeNode>` objects. It uses Python magic to define the conversions.

.. code-block:: python

    from graphtage import IntegerNode, TreeNode
    from graphtage.builder import Builder

    class CustomBuilder(Builder):
        @Builder.builder(int)
        def build_int(self, node: int, children: list[TreeNode]):
            return IntegerNode(node)

>>> CustomBuilder().build_tree(10)
IntegerNode(10)

The :func:`@Builder.builder(int) <graphtage.Builder.builder>` decorator specifies that the function is able to build a Graphtage `TreeNode` object from inputs that are :func:`instanceof` the type `int`. If there are multiple builder functions that match a given object, the function associated with the most specialized type is chosen. For example:

.. code-block:: python

    class Foo:
        pass


    class Bar(Foo):
        pass


    class CustomBuilder(Builder):
        @Builder.builder(Foo)
        def build_foo(self, node: Foo, children: list[TreeNode]):
            return StringNode("foo")

        @Build.builder(Bar)
        def build_bar(self, node: Bar, children: list[TreeNode]):
            return StringNode("bar")

>>> CustomBuilder().build_tree(Foo())
StringNode("foo")
>>> CustomBuilder().build_tree(Bar())
StringNode("bar")

Expanding Children
------------------

So far we have only given examples of the production of leaf nodes, like integers and strings.
What if a node has children, like a list? We can handle this using the :func:`@Builder.expander <graphtage.Builder.expander>` decorator. Here is an example of how a list can be built:

.. code-block:: python

    class CustomBuilder(Builder):
        ...

        @Builder.expander(list)
        def expand_list(self, node: list):
            """Returns an iterable over the node's children"""
            yield from node

        @Builder.builder(list)
        def build_list(self, node: list, children: list[TreeNode]):
            return ListNode(children)

>>> CustomBuilder().build_tree([1, 2, 3, 4])
ListNode([IntegerNode(1), IntegerNode(2), IntegerNode(3), IntegerNode(4)])

If an expander is not defined for a type, it is assumed that the type is a leaf with no children.

If the root node or one of its descendants is of a type that has no associated builder function, a :exc:`NotImplementedError` is raised.

Graphtage has a subclassed builder :class:`graphtage.builder.BasicBuilder` that has builders and expanders for the Python basic types like :class:`int`, :class:`float`, :class:`str`, :class:`bytes`, :class:`list`, :class:`dict`, :class:`set`, and :class:`tuple`. You can extend :class:`graphtage.builder.BasicBuilder` to implement support for additional types.

Custom Nodes
------------

Graphtage provides abstract classes like :class:`graphtage.ContainerNode` and :class:`graphtage.SequenceNode` to aid in the implementation of custom node types. But the easiest way to define a custom node type is to extend off of :class:`graphtage.dataclasses.DataClass`.


.. code-block:: python

    from graphtage import IntegerNode, ListNode, StringNode
    from graphtage.dataclasses import DataClass

    class CustomNode(DataClass):
        name: StringNode
        value: IntegerNode
        attributes: ListNode

This will automatically build a node type that has three children: a string, an integer, and a list.

>>> CustomNode(name=StringNode("the name"), value=IntegerNode(1337), attributes=ListNode((IntegerNode(1), IntegerNode(2), IntegerNode(3))))

Let's say you have another, non-graphtage class that corresponds to :class:`CustomNode`:

.. code-block:: python

    class NonGraphtageClass:
        name: str
        value: int
        attributes: list[int]

You can add support for building Graphtage nodes from this custom class as follows:

.. code-block:: python

    class CustomBuilder(BasicBuilder):
        @Builder.expander(NonGraphtageClass)
        def expand_non_graphtage_class(node: NonGraphtageClass):
            yield node.name
            yield node.value
            yield node.attributes

        @Builder.builder(NonGraphtageClass)
        def build_non_graphtage_class(node: NonGraphtageClass, children: List[TreeNode]) -> CustomNode:
            return CustomNode(*children)
