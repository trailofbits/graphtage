import json
import subprocess
import sys
import tempfile
from io import StringIO
from os.path import join
from typing import List, Tuple
from unittest import TestCase

import graphtage.json
import graphtage.levenshtein
import graphtage.tree
from graphtage import printer
from graphtage.printer import Printer

FROM_OBJ = {"a": 1, "b": [1, 2, 3], "c": {"d": "hello world"}}
TO_OBJ = {"a": 2, "b": [1, 2, 4], "c": {"d": "goodbye world"}}

PROGRESS_BAR = b"Diffing:"


def run_graphtage(*args: str) -> Tuple[bytes, bytes]:
    """Runs the command line in a subprocess and returns what it wrote to stdout and to stderr.

    The subprocess matters: the modules that draw the progress bars resolve the default printer while they are being
    imported, so an in-process test would reuse whichever printer an earlier test installed.

    Args:
        *args: Options to pass before the two file operands.

    Returns:
        Tuple[bytes, bytes]: The bytes written to stdout and to stderr.

    """
    with tempfile.TemporaryDirectory() as tmpdir:
        paths: List[str] = []
        for name, obj in (("from.json", FROM_OBJ), ("to.json", TO_OBJ)):
            path = join(tmpdir, name)
            with open(path, "w") as f:
                f.write(json.dumps(obj))
            paths.append(path)
        command: List[str] = [sys.executable, "-m", "graphtage"]
        command.extend(args)
        command.extend(paths)
        result = subprocess.run(command, capture_output=True)
    if result.returncode not in (0, 1):
        raise AssertionError(f"`graphtage` exited with status {result.returncode}: {result.stderr.decode('utf-8')}")
    return result.stdout, result.stderr


class TestProgress(TestCase):
    def test_progress_is_shown_by_default(self):
        """Without a suppression flag, the diff draws its progress bar on stderr."""
        _, stderr = run_graphtage()
        self.assertIn(PROGRESS_BAR, stderr)

    def test_no_status_suppresses_progress(self):
        """``--no-status`` must leave stderr empty, as :file:`README.md` documents."""
        stdout, stderr = run_graphtage("--no-status")
        self.assertEqual(b"", stderr, f"`--no-status` wrote {len(stderr)} bytes to stderr")
        self.assertIn(b'"a"', stdout)

    def test_quiet_suppresses_progress(self):
        """``--quiet`` must suppress the progress bar along with the log messages."""
        stdout, stderr = run_graphtage("--quiet")
        self.assertEqual(b"", stderr, f"`--quiet` wrote {len(stderr)} bytes to stderr")
        self.assertIn(b'"a"', stdout)

    def test_replacement_printer_reaches_every_progress_bar(self):
        """Every module that draws a progress bar must observe :func:`graphtage.printer.set_default_printer`.

        A module that binds ``DEFAULT_PRINTER`` at import time keeps the printer that was current when it was first
        imported, which is what let the suppression flags be ignored.

        """
        replacement = Printer(out_stream=StringIO(), quiet=True)
        original = printer.get_default_printer()
        printer.set_default_printer(replacement)
        try:
            for module in (graphtage.json, graphtage.levenshtein, graphtage.tree):
                self.assertIs(
                    module.get_default_printer(),
                    replacement,
                    f"{module.__name__} did not observe the replacement printer",
                )
        finally:
            printer.set_default_printer(original)
