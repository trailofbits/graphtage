import unittest
from io import StringIO

import graphtage
from graphtage.printer import Printer
from graphtage.utils import Tempfile
from graphtage.xml import XML


class TestXML(unittest.TestCase):
    def test_infinite_loop(self):
        """Reproduces https://github.com/trailofbits/graphtage/issues/32"""
        xml = XML.default_instance
        one_xml = b"""
<root>
  <parent>
    <child1 attribute1="foo">child1</child1>
    <child2>child2</child2>
  </parent>
</root>
"""
        two_xml = b"""
<root>
  <parent>
    <child1 attribute1="bar">child1</child1>
    <child2>child2</child2>
  </parent>
</root>
"""
        with Tempfile(one_xml) as one, Tempfile(two_xml) as two:
            t1 = xml.build_tree(one)
            t2 = xml.build_tree(two)
            for edit in t1.get_all_edits(t2):
                print(edit)


def build(content: bytes) -> graphtage.TreeNode:
    with Tempfile(content) as path:
        return graphtage.FILETYPES_BY_TYPENAME["xml"].build_tree(path)


def render(node: graphtage.TreeNode) -> str:
    stream = StringIO()
    printer = Printer(out_stream=stream, ansi_color=False)
    graphtage.FILETYPES_BY_TYPENAME["xml"].get_default_formatter().print(printer, node)
    printer.flush(final=True)
    return stream.getvalue()


def render_diff(from_content: bytes, to_content: bytes) -> str:
    return render(build(from_content).diff(build(to_content)))


class TestXMLDiff(unittest.TestCase):
    """Covers XML formatter edit printing, which the unedited round-trip tests cannot reach."""

    def test_inserted_attribute_on_bare_element_is_marked(self):
        """Adding the first attribute used to vanish because if node.attrib skipped an empty DictNode."""
        rendered = render_diff(b"<r><d>x</d></r>", b'<r><d a="1">x</d></r>')
        self.assertIn("a", rendered)
        self.assertIn("++", rendered)
        self.assertIn("1", rendered)

    def test_inserted_child_element_is_marked(self):
        """A new child on an element that already had text used to be omitted from the rendered tree."""
        rendered = render_diff(b"<r><d>x</d></r>", b"<r><d>x<e>y</e></d></r>")
        self.assertIn("<e>", rendered)
        self.assertIn("y", rendered)
        self.assertIn("++", rendered)

    def test_inserted_child_on_empty_element_is_not_self_closed(self):
        """An empty element used to stay self-closing, hiding the inserted subtree."""
        rendered = render_diff(b"<r><d></d></r>", b"<r><d><e>y</e></d></r>")
        self.assertNotIn("<d />", rendered)
        self.assertIn("<e>", rendered)
        self.assertIn("y", rendered)

    def test_adding_an_attribute_when_one_exists_still_works(self):
        rendered = render_diff(b'<r><d a="1">x</d></r>', b'<r><d a="1" b="2">x</d></r>')
        self.assertIn("b", rendered)
        self.assertIn("++", rendered)

    def test_unchanged_empty_element_stays_self_closing(self):
        rendered = render_diff(b"<r><d></d></r>", b"<r><d></d></r>")
        self.assertIn("<d />", rendered)
        self.assertNotIn("++", rendered)
        self.assertNotIn("~~", rendered)
