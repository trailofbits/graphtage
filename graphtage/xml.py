import xml.etree.ElementTree as ET
from typing import Any, Dict, Optional, Iterator, Sequence, Union

from .bounds import Range
from .edits import AbstractCompoundEdit, Insert, Match, Remove
from .graphtage import ContainerNode, DictNode, ListNode, StringNode
from .printer import Printer
from .tree import Edit, EditedTreeNode


class XMLElementEdit(AbstractCompoundEdit):
    def __init__(self, from_node: 'XMLElement', to_node: 'XMLElement'):
        self.tag_edit: Edit = from_node.tag.edits(to_node.tag)
        self.attrib_edit: Edit = from_node.attrib.edits(to_node.attrib)
        if from_node.text is not None and to_node.text is not None:
            self.text_edit: Optional[Edit] = from_node.text.edits(to_node.text)
        elif from_node.text is None:
            self.text_edit: Optional[Edit] = Insert(to_insert=to_node.text, insert_into=from_node)
        elif to_node.text is None:
            self.text_edit: Optional[Edit] = Remove(to_remove=from_node.text, remove_from=from_node)
        else:
            self.text_edit: Optional[Edit] = None
        self.child_edit: Edit = from_node.children.edits(to_node.children)
        super().__init__(
            from_node=from_node,
            to_node=to_node
        )

    def bounds(self) -> Range:
        if self.text_edit is not None:
            text_bounds = self.text_edit.bounds()
        else:
            text_bounds = Range(0, 0)
        return text_bounds + self.tag_edit.bounds() + self.attrib_edit.bounds() + self.child_edit.bounds()

    def edits(self) -> Iterator[Edit]:
        pass

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
        return self.tag_edit.tighten_bounds() or (self.text_edit is not None and self.text_edit.tighten_bounds) or \
            self.attrib_edit.tighten_bounds() or self.child_edit.tighten_bounds()


class XMLElement(ContainerNode):
    def __init__(
            self,
            tag: StringNode,
            attrib: Optional[Dict[StringNode, StringNode]] = None,
            text: Optional[StringNode] = None,
            children: Sequence['XMLElement'] = ()
    ):
        self.tag: StringNode = tag
        if attrib is None:
            attrib = {}
        self.attrib: DictNode = DictNode(attrib)
        self.text: Optional[StringNode] = text
        self.children: ListNode = ListNode(children)

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
        return t_size + self.tag.total_size + self.attrib.total_size + self.children.total_size

    def print(self, printer: Printer):
        pass

    def init_args(self) -> Dict[str, Any]:
        return {
            'tag': self.tag,
            'attrib': {kvp.key: kvp.value for kvp in self.attrib},
            'text': self.text,
            'children': tuple(self.children.__iter__())
        }

    def make_edited(self) -> Union[EditedTreeNode, 'KeyValuePairNode']:
        if self.text is None:
            et = None
        else:
            et = self.text.make_edited()
        return self.edited_type()(
            tag=self.tag.make_edited(),
            attrib={
                kvp.key.make_edited(): kvp.value.make_edited() for kvp in self.attrib
            },
            text=et,
            children=[c.make_edited() for c in self.children]
        )

    def __eq__(self, other):
        if not isinstance(other, XMLElement):
            return False
        return other.tag == self.tag and other.attrib == self.attrib \
            and other.text == self.text and other.children == self.children


def build_tree(path_or_element_tree: Union[str, ET.Element, ET.ElementTree]) -> XMLElement:
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
        children=[build_tree(child) for child in root]
    )
