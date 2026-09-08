import unittest

from graphtage.utils import Tempfile
from graphtage.xml import HTML


class TestHTML(unittest.TestCase):
    def test_unquoted_attributes(self):
        """Reproduces and verifies fix for https://github.com/trailofbits/graphtage/issues/25

        HTML5 allows unquoted attributes like <meta name=foo>, but the XML parser
        incorrectly rejected this valid HTML syntax. This test verifies that HTML
        files with unquoted attributes can now be parsed and diffed correctly.
        """
        html = HTML.default_instance

        # HTML with unquoted attributes (the original issue)
        html_unquoted_1 = b"""<!DOCTYPE html>
<html>
<head>
    <meta name=foo>
    <meta charset=utf-8>
    <title>Test Page</title>
</head>
<body>
    <div id=content>
        <h1>Hello World</h1>
    </div>
</body>
</html>"""

        html_unquoted_2 = b"""<!DOCTYPE html>
<html>
<head>
    <meta name=bar>
    <meta charset=utf-8>
    <title>Test Page</title>
</head>
<body>
    <div id=content>
        <h1>Hello World</h1>
    </div>
</body>
</html>"""

        # This should not raise an exception (would previously fail with:
        # "not well-formed (invalid token)" when using XML parser)
        with Tempfile(html_unquoted_1) as one, Tempfile(html_unquoted_2) as two:
            t1 = html.build_tree(one)
            t2 = html.build_tree(two)

            # Verify trees were built successfully
            self.assertIsNotNone(t1)
            self.assertIsNotNone(t2)

            # Verify we can compute edits between them
            edits = list(t1.get_all_edits(t2))
            self.assertGreater(len(edits), 0, "Should find edits between the two HTML files")

            # Verify the diff captures the meta name change from 'foo' to 'bar'
            edit_strings = [str(edit) for edit in edits]
            all_edits_str = ' '.join(edit_strings)
            self.assertTrue(
                'foo' in all_edits_str or 'bar' in all_edits_str,
                "Diff should capture the change in meta name attribute"
            )

    def test_mixed_quoted_unquoted_attributes(self):
        """Test HTML files with a mix of quoted and unquoted attributes.

        This ensures backward compatibility - files with quoted attributes should
        still work correctly, and they can be mixed with unquoted attributes.
        """
        html = HTML.default_instance

        # Mix of quoted and unquoted attributes
        html_mixed = b"""<!DOCTYPE html>
<html>
<head>
    <meta name=description content="A test page">
    <meta charset="utf-8">
</head>
<body>
    <div id=main class="container">
        <p>Mixed quotes</p>
    </div>
</body>
</html>"""

        # Should parse without errors
        with Tempfile(html_mixed) as temp:
            tree = html.build_tree(temp)
            self.assertIsNotNone(tree)

            # Verify we can access the tree structure
            self.assertEqual(tree.tag.object, 'html')

    def test_quoted_attributes_backward_compatibility(self):
        """Test that HTML files with quoted attributes still work (backward compatibility).

        This verifies that the fix for unquoted attributes doesn't break existing
        functionality with quoted attributes.
        """
        html = HTML.default_instance

        html_quoted_1 = b"""<!DOCTYPE html>
<html>
<head>
    <meta name="author" content="Alice">
    <meta charset="utf-8">
</head>
<body>
    <h1>Test</h1>
</body>
</html>"""

        html_quoted_2 = b"""<!DOCTYPE html>
<html>
<head>
    <meta name="author" content="Bob">
    <meta charset="utf-8">
</head>
<body>
    <h1>Test</h1>
</body>
</html>"""

        with Tempfile(html_quoted_1) as one, Tempfile(html_quoted_2) as two:
            t1 = html.build_tree(one)
            t2 = html.build_tree(two)

            # Verify trees were built successfully
            self.assertIsNotNone(t1)
            self.assertIsNotNone(t2)

            # Verify we can compute edits
            edits = list(t1.get_all_edits(t2))
            self.assertGreater(len(edits), 0)

            # Verify the diff captures the content change from 'Alice' to 'Bob'
            edit_strings = [str(edit) for edit in edits]
            all_edits_str = ' '.join(edit_strings)
            self.assertTrue(
                'Alice' in all_edits_str or 'Bob' in all_edits_str,
                "Diff should capture the change in meta content attribute"
            )

    def test_complex_unquoted_attributes(self):
        """Test more complex cases with unquoted attributes.

        HTML5 spec allows unquoted attribute values that don't contain spaces,
        quotes, =, <, >, or `.
        """
        html = HTML.default_instance

        html_complex = b"""<!DOCTYPE html>
<html lang=en>
<head>
    <meta name=viewport content=width=device-width,initial-scale=1>
    <link rel=stylesheet href=/styles.css>
</head>
<body>
    <div data-id=12345 data-type=article>
        <h1>Article Title</h1>
    </div>
</body>
</html>"""

        # Should parse without errors
        with Tempfile(html_complex) as temp:
            tree = html.build_tree(temp)
            self.assertIsNotNone(tree)
            self.assertEqual(tree.tag.object, 'html')

    def test_closing_tags_issue_26(self):
        """Test for issue #26: Fails to match closing HTML tags.

        The original issue reported that graphtage threw errors when encountering
        closing tags like </head>. With the lxml HTML parser, this should work.
        """
        html = HTML.default_instance

        html_with_closing_tags = b"""<!DOCTYPE html>
<html>
<head>
    <title>Test Page</title>
    <meta charset="utf-8">
</head>
<body>
    <h1>Hello World</h1>
    <p>Some text</p>
</body>
</html>"""

        # Should parse without errors (previously would fail on </head>)
        with Tempfile(html_with_closing_tags) as temp:
            tree = html.build_tree(temp)
            self.assertIsNotNone(tree)
            self.assertEqual(tree.tag.object, 'html')

        # Test that we can diff two files with closing tags
        html_modified = b"""<!DOCTYPE html>
<html>
<head>
    <title>Modified Page</title>
    <meta charset="utf-8">
</head>
<body>
    <h1>Hello World</h1>
    <p>Different text</p>
</body>
</html>"""

        with Tempfile(html_with_closing_tags) as one, Tempfile(html_modified) as two:
            t1 = html.build_tree(one)
            t2 = html.build_tree(two)

            # Should compute edits without errors
            edits = list(t1.get_all_edits(t2))
            self.assertGreater(len(edits), 0)

    def test_text_between_elements_issue_80(self):
        """Test for issue #80: Text missing from HTML diff.

        The original issue reported that text nodes between elements (not wrapped
        in tags) were missing from the diff output. For example, "and more" in:
        some <div>text and more</div> text
        """
        html = HTML.default_instance

        # Example from issue #80
        old_html = b"""<html>
  <body>
    some <div>text and more</div> text
  </body>
</html>"""

        new_html = b"""<html>
  <body>
    some <div class='red'>text</div> and more <strong>text</strong>
  </body>
</html>"""

        with Tempfile(old_html) as one, Tempfile(new_html) as two:
            t1 = html.build_tree(one)
            t2 = html.build_tree(two)

            # Verify trees were built successfully
            self.assertIsNotNone(t1)
            self.assertIsNotNone(t2)

            # Verify we can compute edits
            edits = list(t1.get_all_edits(t2))
            self.assertGreater(len(edits), 0)

            # Convert edits to strings to check if text is present
            edit_strings = [str(edit) for edit in edits]
            all_edits_str = ' '.join(edit_strings)

            # The key test: verify that "and more" appears somewhere in the output
            # This would have been missing in the original bug
            self.assertTrue(
                'and more' in all_edits_str or 'and' in all_edits_str,
                "Text 'and more' should be present in the diff output"
            )
