import json
import unittest
import xml.etree.ElementTree as ET
from io import StringIO

from graphtage.edits import Insert, Match, Remove
from graphtage.graphtage import BuildOptions, StringNode
from graphtage.json import JSONFormatter
from graphtage.printer import Printer
from graphtage.utils import Tempfile
from graphtage.xml import HTML, XML, XMLElement, XMLElementEdit, XMLFormatter, build_tree


def render(node, formatter=XMLFormatter.DEFAULT_INSTANCE):
    stream = StringIO()
    printer = Printer(out_stream=stream, ansi_color=False)
    formatter.print(printer, node)
    printer.flush(final=True)
    return stream.getvalue()


def build(content):
    return build_tree(ET.fromstring(content))


class TestXMLTail(unittest.TestCase):
    def test_tail_only_change_is_not_a_match(self):
        left = build('<r><a>x</a><b>y</b></r>')
        right = build('<r><a>x</a>TAIL_ONLY_NEW<b>y</b></r>')
        self.assertNotEqual(left, right)
        self.assertNotIsInstance(left.edits(right), Match)
        self.assertIn('++TAIL_ONLY_NEW++', render(left.diff(right)))

    def test_tail_edits_in_both_formats(self):
        for old, new, expected in (
            ('', 'INSERTED_TAIL', '++INSERTED_TAIL++'),
            ('REMOVED_TAIL', '', '~~REMOVED_TAIL~~'),
            ('old', 'new', '~~old~~++new++'),
            ('same', 'same', 'same'),
        ):
            for formatter in (XMLFormatter.DEFAULT_INSTANCE, JSONFormatter.DEFAULT_INSTANCE):
                with self.subTest(old=old, new=new, formatter=type(formatter).__name__):
                    left = build(f'<r><a />{old}<b /></r>')
                    right = build(f'<r><a />{new}<b /></r>')
                    rendered = render(left.diff(right), formatter)
                    formatted_expected = expected
                    if formatter is JSONFormatter.DEFAULT_INSTANCE and not old:
                        formatted_expected = f'++"{new}"++'
                    elif formatter is JSONFormatter.DEFAULT_INSTANCE and not new:
                        formatted_expected = f'~~"{old}"~~'
                    self.assertIn(formatted_expected, rendered)
                    if old == new:
                        self.assertNotIn('++', rendered)
                        self.assertNotIn('~~', rendered)
                    if formatter is XMLFormatter.DEFAULT_INSTANCE:
                        self.assertLess(rendered.index('<a />'), rendered.index(expected))
                        self.assertLess(rendered.index(expected), rendered.index('<b />'))

    def test_mixed_content_round_trips(self):
        samples = (
            '<r>before<a>x</a> between <b>y</b> after</r>',
            '<r><a /> &amp; &lt; &gt; café Ω 😀 <b />end</r>',
            '<r>\n  <a />\n  <b />\n</r>',
            '<r><a>inner<b />nested</a>outer<c />last</r>',
        )
        for content in samples:
            with self.subTest(content=content):
                tree = build(content)
                expected = ET.tostring(ET.fromstring(content), encoding='unicode')
                self.assertEqual(str(tree.to_obj()), expected)
                self.assertEqual(render(tree), expected)
                self.assertEqual(render(tree.diff(build(content))), expected)
                self.assertEqual(list(ET.fromstring(render(tree)).itertext()), list(ET.fromstring(content).itertext()))

    def test_tail_in_model_traversal_size_and_copy(self):
        for has_text in (False, True):
            for has_tail in (False, True):
                with self.subTest(has_text=has_text, has_tail=has_tail):
                    text = 'text' if has_text else ''
                    tail = 'tail' if has_tail else ''
                    tree = build(f'<r><a key="value">{text}<b /></a>{tail}</r>')
                    child = tree._children._children[0]
                    self.assertEqual(child.to_obj().tail, tail or None)
                    if has_tail:
                        self.assertIs(child.tail.parent, child)
                        self.assertIn(child.tail, child.children())
                        self.assertTrue(any(node is child.tail for node in tree.dfs()))
                    without_tail = build(f'<r><a key="value">{text}<b /></a></r>')
                    self.assertEqual(tree.total_size - without_tail.total_size, len(tail))
                    copied = tree.copy()
                    self.assertEqual(copied, tree)
                    self.assertEqual(hash(copied), hash(tree))
                    self.assertEqual(str(copied), str(tree))
                    self.assertTrue(all(a is not b for a, b in zip(tree.dfs(), copied.dfs(), strict=True)))
                    self.assertIs(copied._children._children[0].parent, copied._children)

    def test_whitespace_in_tail_is_significant(self):
        for old, new in (('', ' '), (' tail', 'tail '), ('\n', '\n\n')):
            with self.subTest(old=old, new=new):
                left = build(f'<r><a />{old}<b /></r>')
                right = build(f'<r><a />{new}<b /></r>')
                self.assertNotEqual(left, right)
                self.assertGreater(left.diff(right).edit.bounds().lower_bound, 0)

    def test_tail_edit_cost_and_completion(self):
        for old, new, edit_type in (('', 'tail', Insert), ('tail', '', Remove), ('old', 'new', None)):
            with self.subTest(old=old, new=new):
                left = build(f'<r><a />{old}</r>')._children._children[0]
                right = build(f'<r><a />{new}</r>')._children._children[0]
                edit = left.make_edited().edits(right)
                self.assertIsInstance(edit, XMLElementEdit)
                if edit_type is not None:
                    self.assertIsInstance(edit.tail_edit, edit_type)
                self.assertIn(edit.tail_edit, list(edit.edits()))
                while edit.tighten_bounds():
                    pass
                self.assertTrue(edit.is_complete())
                self.assertTrue(edit.bounds().definitive())
                self.assertEqual(edit.bounds(), edit.tail_edit.bounds())

    def test_json_tail_view_preserves_tree_and_edits(self):
        tree = build('<r><a>x</a>TAIL_ONLY_HERE<b />last</r>')
        parents = [node.parent for node in tree.dfs()]
        view = json.loads(render(tree, JSONFormatter.DEFAULT_INSTANCE))
        self.assertEqual(view['children'][0]['tail'], 'TAIL_ONLY_HERE')
        self.assertEqual(view['children'][1]['tail'], 'last')
        self.assertNotIn('tail', view)
        self.assertEqual([node.parent for node in tree.dfs()], parents)
        self.assertEqual(json.loads(render(tree, JSONFormatter.DEFAULT_INSTANCE)), view)
        diff = build('<r><a>x</a><b /></r>').diff(tree)
        parents = [node.parent for node in diff.dfs()]
        first_render = render(diff, JSONFormatter.DEFAULT_INSTANCE)
        self.assertIn('++"TAIL_ONLY_HERE"++', first_render)
        self.assertEqual(render(diff, JSONFormatter.DEFAULT_INSTANCE), first_render)
        self.assertTrue(all(node.parent is parent for node, parent in zip(diff.dfs(), parents, strict=True)))

    def test_tail_on_inserted_and_removed_children(self):
        old = build('<r><a>x</a>stay</r>')
        new = build('<r><a>x</a>stay<b>y</b>TAIL_OF_NEW_CHILD</r>')
        for formatter in (XMLFormatter.DEFAULT_INSTANCE, JSONFormatter.DEFAULT_INSTANCE):
            with self.subTest(formatter=type(formatter).__name__):
                self.assertIn('TAIL_OF_NEW_CHILD', render(old.diff(new), formatter))
                self.assertIn('TAIL_OF_NEW_CHILD', render(new.diff(old), formatter))

    def test_xml_and_html_filetypes_preserve_issue_80_tail(self):
        old = b'<html><body>some <div>text and more</div> text</body></html>'
        new = b'<html><body>some <div class="red">text</div> and more <strong>text</strong></body></html>'
        for filetype in (XML.default_instance, HTML.default_instance):
            with self.subTest(filetype=filetype):
                with Tempfile(old) as old_path, Tempfile(new) as new_path:
                    left = filetype.build_tree(old_path)
                    right = filetype.build_tree(new_path)
                self.assertEqual(right._children._children[0]._children._children[0].tail.object, ' and more ')
                # The old div's .text cannot satisfy this assertion: render only the new tree.
                self.assertIn('</div> and more <strong>', render(right))
                diff = render(left.diff(right))
                self.assertIn('</div> ++and ++~~text~~++more ++', diff)

    def test_root_tail_and_constructor_compatibility(self):
        element = ET.Element('a')
        element.tail = 'outside & < >\n'
        tree = build_tree(element)
        self.assertEqual(str(tree), '<a />outside &amp; &lt; &gt;\n')
        self.assertEqual(render(tree), str(tree))
        self.assertEqual(tree.copy(), tree)
        tree = XMLElement(StringNode('a'), {}, None, (), False, False)
        self.assertIsNone(tree.tail)
        with Tempfile(b'<r><a />tail</r>') as path:
            tree = XML.default_instance.build_tree(path, BuildOptions(allow_key_edits=False, auto_match_keys=False))
        self.assertEqual(json.loads(render(tree, JSONFormatter.DEFAULT_INSTANCE))['children'][0]['tail'], 'tail')
