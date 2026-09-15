"""Regression tests for #128.

Importing graphtage.printer used to have the side effect of calling
colorama.init() at module scope, because NULL_PRINTER (constructed at
import time) was built on top of a NullWriter whose isatty() incorrectly
returned True. colorama.init() replaces sys.stdout/sys.stderr with a
stripping wrapper, so any later, real Printer that writes to a
redirected/piped stream had its ANSI escapes silently stripped even when
color was explicitly forced with --color.
"""

import importlib
import subprocess
import sys
from unittest import TestCase

from graphtage.printer import NullWriter, NULL_PRINTER, Printer


class TestNullWriter(TestCase):
    def test_isatty_is_false(self):
        """A writer that discards everything has no terminal to color."""
        self.assertFalse(NullWriter().isatty())

    def test_null_printer_does_not_enable_color(self):
        self.assertFalse(NULL_PRINTER.ansi_color)


class TestImportDoesNotMutateStdStreams(TestCase):
    def test_importing_printer_module_does_not_wrap_stdout(self):
        """Reproduces #128 in a fresh subprocess so a prior import in this
        test process (or colorama's own global state) can't mask the bug.
        """
        script = (
            "import sys\n"
            "before = sys.stdout\n"
            "import graphtage.printer\n"
            "after = sys.stdout\n"
            "assert before is after, (type(before), type(after))\n"
            "print('OK')\n"
        )
        result = subprocess.run(
            [sys.executable, "-c", script],
            capture_output=True,
            text=True,
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(result.stdout.strip(), "OK")


class TestForcedColorSurvivesRedirection(TestCase):
    def test_color_flag_emits_ansi_escapes_when_redirected(self):
        """End-to-end reproduction of the issue's own repro steps: forcing
        --color on output redirected to a file (i.e. not a tty) must still
        emit ANSI escape sequences.
        """
        import json
        import os
        import tempfile

        with tempfile.TemporaryDirectory() as tmpdir:
            a_path = os.path.join(tmpdir, "a.json")
            b_path = os.path.join(tmpdir, "b.json")
            with open(a_path, "w") as f:
                json.dump({"a": 1}, f)
            with open(b_path, "w") as f:
                json.dump({"a": 2}, f)

            result = subprocess.run(
                [sys.executable, "-m", "graphtage", "--no-status", "--color", a_path, b_path],
                capture_output=True,
                text=True,
            )
            self.assertIn("\x1b", result.stdout, "expected ANSI escapes in forced-color redirected output")
