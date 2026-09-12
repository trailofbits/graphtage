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


def _render_with_formatter(tree: graphtage.TreeNode, formatter) -> str:
    stream = StringIO()
    printer = Printer(out_stream=stream, ansi_color=False)
    formatter.print(printer, tree)
    printer.flush(final=True)
    return stream.getvalue()


class TestXMLCrossFormat(unittest.TestCase):
    """Covers https://github.com/trailofbits/graphtage/issues/187."""

    XML_SAMPLE = b"<r><d>x</d></r>"
    HTML_SAMPLE = b"<html><body>x</body></html>"
    OUTPUT_FORMATS = ("xml", "html", "json", "json5", "yaml", "toml", "csv", "ini", "plist")

    def _build(self, typename: str, content: bytes) -> graphtage.TreeNode:
        with Tempfile(content) as path:
            return graphtage.FILETYPES_BY_TYPENAME[typename].build_tree(path)

    def test_xml_to_json_does_not_raise(self):
        tree = self._build("xml", self.XML_SAMPLE)
        rendered = _render_with_formatter(
            tree.diff(tree),
            graphtage.FILETYPES_BY_TYPENAME["json"].get_default_formatter(),
        )
        self.assertIn("tag", rendered)
        self.assertIn("r", rendered)

    def test_identical_xml_through_every_formatter(self):
        tree = self._build("xml", self.XML_SAMPLE)
        diff = tree.diff(self._build("xml", self.XML_SAMPLE))
        for name in self.OUTPUT_FORMATS:
            with self.subTest(format=name):
                formatter = graphtage.FILETYPES_BY_TYPENAME[name].get_default_formatter()
                rendered = _render_with_formatter(diff, formatter)
                self.assertIsInstance(rendered, str)

    def test_identical_html_through_every_formatter(self):
        tree = self._build("html", self.HTML_SAMPLE)
        diff = tree.diff(self._build("html", self.HTML_SAMPLE))
        for name in self.OUTPUT_FORMATS:
            with self.subTest(format=name):
                formatter = graphtage.FILETYPES_BY_TYPENAME[name].get_default_formatter()
                rendered = _render_with_formatter(diff, formatter)
                self.assertIsInstance(rendered, str)

    def test_xml_diff_through_json_does_not_raise(self):
        left = self._build("xml", self.XML_SAMPLE)
        right = self._build("xml", b"<r><d>y</d></r>")
        rendered = _render_with_formatter(
            left.diff(right),
            graphtage.FILETYPES_BY_TYPENAME["json"].get_default_formatter(),
        )
        self.assertIn("tag", rendered)
