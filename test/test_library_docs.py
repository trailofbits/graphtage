"""Execute the programmatic diff guide against the public API."""

import doctest
from pathlib import Path

from graphtage import printer


def test_consuming_diffs_examples():
    """Keep the published examples, outputs, and edge cases in sync with the implementation."""
    path = Path(__file__).resolve().parents[1] / "docs" / "library.rst"
    text = path.read_text(encoding="utf-8")
    guide = text.split(".. _consuming-diffs:\n", 1)[1].split("\nDiffing Python Source Code\n", 1)[0]
    examples = doctest.DocTestParser().get_doctest(guide, {}, "consuming-diffs", str(path), 0)
    original_printer = printer.get_default_printer()
    printer.set_default_printer(printer.NULL_PRINTER)
    try:
        result = doctest.DocTestRunner().run(examples)
    finally:
        printer.set_default_printer(original_printer)
    assert result.attempted > 0
    assert result.failed == 0
