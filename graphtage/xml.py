"""A :class:`graphtage.Filetype` for parsing, diffing, and rendering XML files.

This class is also currently used for parsing HTML.

The parser is implemented atop :mod:`xml.etree.ElementTree`. Any XML or HTML accepted by that module will also be
accepted by this module.

"""

import html
import os
import xml.etree.ElementTree as ET
from typing import Collection, Dict, Optional, Iterator, Sequence, Union

from .bounds import Range
from .edits import AbstractCompoundEdit, Insert, Match, Remove
from .graphtage import BuildOptions, ContainerNode, DictNode, Filetype, FixedKeyDictNode, KeyValuePairNode, LeafNode, \
    ListNode, StringFormatter, StringNode
from .json import JSONFormatter
from .printer import Printer
from .sequences import SequenceFormatter
from .tree import Edit, EditedTreeNode, GraphtageFormatter, TreeNode


class XMLElementEdit(AbstractCompoundEdit):
    """An edit on an XML element."""
    def __init__(self, from_node: 'XMLElement', to_node: 'XMLElement'):
        """Initializes an XML element edit.

        Args:
            from_node: The node being edited.
            to_node: The node to which :obj:`from_node` will be transformed.
        """
        self.tag_edit: Edit = from_node.tag.edits(to_node.tag)
        """The edit to transform this element's tag."""
        self.attrib_edit: Edit = from_node.attrib.edits(to_node.attrib)
        """The edit to transform this element's attributes."""
        if from_node.text is not None and to_node.text is not None:
            self.text_edit: Optional[Edit] = from_node.text.edits(to_node.text)
            """The edit to transform this element's text."""
        elif from_node.text is None and to_node.text is not None:
            self.text_edit: Optional[Edit] = Insert(to_insert=to_node.text, insert_into=from_node)
        elif to_node.text is None and from_node.text is not None:
            self.text_edit: Optional[Edit] = Remove(to_remove=from_node.text, remove_from=from_node)
        else:
            self.text_edit: Optional[Edit] = None
        self.child_edit: Edit = from_node._children.edits(to_node._children)
        """The edit to transform this node's children."""
        super().__init__(
            from_node=from_node,
            to_node=to_node
        )

    def print(self, formatter: GraphtageFormatter, printer: Printer):
        formatter.get_formatter(self.from_node)(printer, self.from_node)

    def bounds(self) -> Range:
        if self.text_edit is not None:
            text_bounds = self.text_edit.bounds()
        else:
            text_bounds = Range(0, 0)
        return text_bounds + self.tag_edit.bounds() + self.attrib_edit.bounds() + self.child_edit.bounds()

    def edits(self) -> Iterator[Edit]:
        yield self.tag_edit
        yield self.attrib_edit
        if self.text_edit is not None:
            yield self.text_edit
        yield self.child_edit

    def is_complete(self) -> bool:
        return self.tag_edit.is_complete() and (self.text_edit is None or self.text_edit.is_complete()) \
            and self.attrib_edit.is_complete() and self.child_edit.is_complete()

    def tighten_bounds(self) -> bool:
        if self.tag_edit.tighten_bounds():
            return True
        elif self.text_edit is not None and self.text_edit.tighten_bounds():
            return True
        elif self.attrib_edit.tighten_bounds():
            return True
        elif self.child_edit.tighten_bounds():
            return True
        else:
            return False


class XMLElementObj:
    """An object for interacting with :class:`XMLElement` from command line expressions."""
    def __init__(
            self,
            tag: str,
            attrib: Dict[str, str],
            text: Optional[str] = None,
            children: Optional[Sequence['XMLElementObj']] = ()
    ):
        """Initializes an XML Element Object.

        Args:
            tag: The tag of the element.
            attrib: The attributes of the element.
            text: The text of the element.
            children: The children of the element.
        """
        self.tag: str = tag
        """The tag of this element."""
        self.attrib: Dict[str, str] = attrib
        """The attributes of this element."""
        self.text: Optional[str] = text
        """The text of this element."""
        self.children: Optional[Sequence['XMLElementObj']] = children
        """The children of this element."""

    def __repr__(self):
        return f"{self.__class__.__name__}(tag={self.tag!r}, attrib={self.attrib!r}, text={self.text!r}, children={self.children!r})"

    def __str__(self):
        ret = f'<{self.tag}'
        for k, v in self.attrib.items():
            val = html.escape(v).replace('"', '\\"')
            ret = f"{ret} {k!s}=\"{val!s}\""
        if not self.text and not self.children:
            return f"{ret} />"
        ret = f"{ret}>"
        if self.text is not None:
            ret = f"{ret}{html.escape(self.text)!s}"
        if self.children is not None:
            for child in self.children:
                ret = f"{ret}{child!s}"
        return f"{ret}</{self.tag}>"


class XMLElement(ContainerNode):
    """"A node representing an XML element."""

    def __init__(
            self,
            tag: StringNode,
            attrib: Optional[Dict[StringNode, StringNode]] = None,
            text: Optional[StringNode] = None,
            children: Sequence['XMLElement'] = (),
            allow_key_edits: bool = True
    ):
        """Initializes an XML element.

        Args:
            tag: The tag of the element.
            attrib: The attributes of the element.
            text: The text of the element.
            children: The children of the element.
            allow_key_edits: Whether or not to allow keys to be edited when matching element attributes.
        """
        self.tag: StringNode = tag
        """The tag of this element."""
        tag.quoted = False
        if attrib is None:
            attrib = {}
        if allow_key_edits:
            self.attrib: DictNode = DictNode.from_dict(attrib)
            """The attributes of this element."""
        else:
            self.attrib = FixedKeyDictNode.from_dict(attrib)
        if isinstance(self, EditedTreeNode):
            self.attrib = self.attrib.make_edited()
        for key, _ in self.attrib.items():
            key.quoted = False
        self.text: Optional[StringNode] = text
        """The text of this element."""
        if self.text is not None:
            self.text.quoted = False
        self._children: ListNode = ListNode(children)
        if isinstance(self, EditedTreeNode):
            self._children = self._children.make_edited()

    def to_obj(self):
        if self.text is None:
            text_obj = None
        else:
            text_obj = self.text.to_obj()
        return XMLElementObj(
            tag=self.tag.to_obj(),
            attrib=self.attrib.to_obj(),
            text=text_obj,
            children=self._children.to_obj()
        )

    def children(self) -> Collection[TreeNode]:
        ret = (self.tag, self.attrib)
        if self.text is not None:
            return ret + (self.text, self._children)
        else:
            return ret + (self._children,)

    def __iter__(self) -> Iterator[TreeNode]:
        return iter(self.children())

    def __len__(self) -> int:
        return len(self.children())

    def __repr__(self):
        return f"{self.__class__.__name__}(tag={self.tag!r}, attrib={self.attrib!r}, text={self.text!r}, children={self._children!r})"

    def __str__(self):
        return str(self.to_obj())

    def __hash__(self):
        return hash(self.children())

    def edits(self, node) -> Edit:
        if self == node:
            return Match(self, node, 0)
        else:
            return XMLElementEdit(self, node)

    def calculate_total_size(self) -> int:
        if self.text is None:
            t_size = 0
        else:
            t_size = self.text.total_size
        return t_size + self.tag.total_size + self.attrib.total_size + self._children.total_size

    def __eq__(self, other):
        if not isinstance(other, XMLElement):
            return False
        my_text = self.text
        if my_text is not None:
            my_text = my_text.object.strip()
        else:
            my_text = ''
        other_text = other.text
        if other_text is not None:
            other_text = other_text.object.strip()
        else:
            other_text = ''
        return other.tag == self.tag and other.attrib == self.attrib \
               and other_text == my_text and other._children == self._children

    def print(self, printer: Printer):
        return XMLFormatter.DEFAULT_INSTANCE.print(printer, self)


def build_tree(
        path_or_element_tree: Union[str, ET.Element, ET.ElementTree],
        options: Optional[BuildOptions] = None
) -> XMLElement:
    """Constructs an XML element node from an XML file."""
    if isinstance(path_or_element_tree, ET.Element):
        root: ET.Element = path_or_element_tree
    else:
        if isinstance(path_or_element_tree, str):
            tree: ET.ElementTree = ET.parse(path_or_element_tree)
        else:
            tree: ET.ElementTree = path_or_element_tree
        root: ET.Element = tree.getroot()
    if root.text:
        text = StringNode(root.text)
    else:
        text = None
    return XMLElement(
        tag=StringNode(root.tag),
        attrib={
            StringNode(k): StringNode(v) for k, v in root.attrib.items()
        },
        text=text,
        children=[build_tree(child, options) for child in root],
        allow_key_edits=options is None or options.allow_key_edits
    )


class XMLChildFormatter(SequenceFormatter):
    is_partial = True

    def __init__(self):
        super().__init__('', '', '')

    def item_newline(self, printer: Printer, is_first: bool = False, is_last: bool = False):
        if not is_first:
            printer.newline()

    def print_ListNode(self, *args, **kwargs):
        super().print_SequenceNode(*args, **kwargs)


class XMLElementAttribFormatter(SequenceFormatter):
    is_partial = True

    def __init__(self):
        super().__init__('', '', '')

    def item_newline(self, printer: Printer, is_first: bool = False, is_last: bool = False):
        pass

    def print_MultiSetNode(self, *args, **kwargs):
        self.print_SequenceNode(*args, **kwargs)

    def print_MappingNode(self, *args, **kwargs):
        self.print_SequenceNode(*args, **kwargs)

    def print_KeyValuePairNode(self, printer: Printer, node: KeyValuePairNode):
        printer.write(' ')
        node.key.quoted = False
        self.print(printer, node.key)
        printer.write('=')
        node.value.quoted = True
        self.print(printer, node.value)


class XMLStringFormatter(StringFormatter):
    is_partial = True

    def escape(self, c: str) -> str:
        """String escape.

        This function is called once for each character in the string.

        Returns:
            str: The escaped version of `c`, or `c` itself if no escaping is required.

        This is equivalent to::

            html.escape(c)

        """
        return html.escape(c)

    def write_char(self, printer: Printer, c: str, index: int, num_edits: int, removed=False, inserted=False):
        if c != '\n' or index < num_edits - 1:
            super().write_char(printer, c, index, num_edits, removed, inserted)


class XMLFormatter(GraphtageFormatter):
    sub_format_types = [XMLStringFormatter, XMLChildFormatter, XMLElementAttribFormatter]

    def _print_text(self, element: XMLElement, printer: Printer):
        if element.text is None:
            return
        elif element.text.edited and element.text.edit is not None and element.text.edit.bounds().lower_bound > 0:
            self.print(printer, element.text.edit)
            return
        text = element.text.object.strip()
        if '\n' not in text and not element._children._children:
            printer.write(html.escape(text))
            return
        with printer.indent():
            sections = text.split('\n')
            if not sections[0]:
                sections = sections[1:]
            for section in sections:
                printer.write(html.escape(section))
                printer.newline()

    def print_LeafNode(self, printer: Printer, node: LeafNode):
        printer.write(html.escape(str(node.object)))

    def print_XMLElement(self, printer: Printer, node: XMLElement):
        printer.write('<')
        self.print(printer, node.tag)
        if node.attrib:
            self.print(printer, node.attrib)
        if node._children._children or (node.text is not None and '\n' in node.text.object):
            printer.write('>')
            if node.text is not None:
                self.print(printer, node.text)
            self.print(printer, node._children)
            printer.write('</')
            self.print(printer, node.tag)
            printer.write('>')
        elif node.text is not None:
            printer.write('>')
            self.print(printer, node.text)
            printer.write('</')
            self.print(printer, node.tag)
            printer.write('>')
        else:
            printer.write(' />')


class XML(Filetype):
    """The XML file type."""
    def __init__(self):
        """Initializes the XML file type.

        By default, XML associates itself with the "xml", "application/xml", and "text/xml" MIME types.

        """
        super().__init__(
            'xml',
            'application/xml',
            'text/xml'
        )

    def build_tree(self, path: str, options: Optional[BuildOptions] = None) -> TreeNode:
        return build_tree(path, options)

    def build_tree_handling_errors(self, path: str, options: Optional[BuildOptions] = None) -> Union[str, TreeNode]:
        try:
            return self.build_tree(path=path, options=options)
        except ET.ParseError as pe:
            return f'Error parsing {os.path.basename(path)}: {pe.msg}'

    def get_default_formatter(self) -> XMLFormatter:
        return XMLFormatter.DEFAULT_INSTANCE


class HTML(XML):
    """The HTML file type."""
    def __init__(self):
        """Initializes the HTML file type.

        By default, HTML associates itself with the "html", "text/html", and "application/xhtml+xml" MIME types.

        """
        Filetype.__init__(
            self,
            'html',
            'text/html',
            'application/xhtml+xml'
        )


# Tell JSON how to format XML:
def _json_print_XMLElement(self: JSONFormatter, printer: Printer, node: XMLElement):
    kvps = [
        KeyValuePairNode(StringNode('tag'), node.tag),
    ]
    if len(node.attrib) > 0:
        kvps.append(KeyValuePairNode(StringNode('attrs'), node.attrib))
    if node.text is not None:
        kvps.append(KeyValuePairNode(StringNode('text'), node.text))
    kvps.append(KeyValuePairNode(StringNode('children'), node._children))
    self.print(printer, DictNode(kvps))


setattr(JSONFormatter, "print_XMLElement", _json_print_XMLElement)
