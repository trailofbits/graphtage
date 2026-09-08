import subprocess
import sys
import tempfile
from os.path import join
from unittest import TestCase

FROM_JSON = '{"a": 1, "b": [1, 2, 3]}'
TO_JSON = '{"a": 2, "b": [1, 2, 4]}'

ANSI_ESCAPE = b"\x1b["


def run_graphtage(*args: str) -> bytes:
    """Runs the command line with its output redirected to a pipe, and returns what was written to stdout."""
    with tempfile.TemporaryDirectory() as tmpdir:
        from_path = join(tmpdir, "from.json")
        to_path = join(tmpdir, "to.json")
        for path, contents in ((from_path, FROM_JSON), (to_path, TO_JSON)):
            with open(path, "w") as f:
                f.write(contents)
        command: list[str] = [sys.executable, "-m", "graphtage", "--no-status"]
        command.extend(args)
        command.extend((from_path, to_path))
        result = subprocess.run(command, capture_output=True)
    if result.returncode not in (0, 1):
        raise AssertionError(f"`graphtage` exited with status {result.returncode}: {result.stderr.decode('utf-8')}")
    return result.stdout


class TestPrinter(TestCase):
    def test_import_does_not_wrap_stdout(self):
        """Importing the library must not replace :attr:`sys.stdout` with colorama's wrapper."""
        code = (
            "import sys\n"
            "before = type(sys.stdout).__name__\n"
            "import graphtage\n"
            "with open(sys.argv[1], 'w') as f:\n"
            "    f.write(f'{before} {type(sys.stdout).__name__}')\n"
        )
        with tempfile.TemporaryDirectory() as tmpdir:
            out_path = join(tmpdir, "stdout_types.txt")
            subprocess.run([sys.executable, "-c", code, out_path], capture_output=True, check=True)
            with open(out_path) as f:
                before, after = f.read().split()
        self.assertEqual(before, after, "`import graphtage` replaced sys.stdout")

    def test_forced_color_is_not_stripped_when_redirected(self):
        """``--color`` must emit ANSI escapes even though stdout is a pipe rather than a terminal."""
        self.assertIn(ANSI_ESCAPE, run_graphtage("--color"))

    def test_redirected_output_is_uncolored_by_default(self):
        """Without ``--color``, a redirected diff must stay free of ANSI escapes."""
        self.assertNotIn(ANSI_ESCAPE, run_graphtage())

    def test_html_output_is_colored_when_forced(self):
        """``--html --color`` must emit HTML colors rather than ANSI escapes."""
        output = run_graphtage("--html", "--color")
        self.assertIn(b"color:", output)
        self.assertNotIn(ANSI_ESCAPE, output)
