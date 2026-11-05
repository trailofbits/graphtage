import contextlib
import io
import os
import shutil
import tempfile
import unittest

from graphtage import version
from graphtage.__main__ import EXIT_DIFFERENCES_FOUND, EXIT_ERROR, EXIT_SUCCESS, main


class TestCommandLine(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.mkdtemp()
        self.addCleanup(shutil.rmtree, self.directory, True)

    def write(self, name, contents):
        path = os.path.join(self.directory, name)
        with open(path, 'w') as f:
            f.write(contents)
        return path

    def run_graphtage(self, *arguments):
        """Runs the command line interface, returning its exit status and everything it wrote to STDOUT."""
        out = io.StringIO()
        with contextlib.redirect_stdout(out), contextlib.redirect_stderr(io.StringIO()):
            status = main(['graphtage', '--no-status', *arguments])
        return status, out.getvalue()

    def test_to_mime_is_honored(self):
        """`--to-mime` used to be discarded in favor of `--from-mime`, so the second file's type was guessed."""
        from_path = self.write('one.json', '{"a": 1}\n')
        to_path = self.write('two_without_an_extension', 'a: 2\n')
        status, output = self.run_graphtage('--to-mime', 'application/x-yaml', from_path, to_path)
        self.assertEqual(EXIT_DIFFERENCES_FOUND, status)
        self.assertIn('2', output)

    def test_from_mime_is_honored(self):
        from_path = self.write('one_without_an_extension', 'a: 1\n')
        to_path = self.write('two.json', '{"a": 2}\n')
        status, output = self.run_graphtage('--from-mime', 'application/x-yaml', from_path, to_path)
        self.assertEqual(EXIT_DIFFERENCES_FOUND, status)
        self.assertIn('2', output)

    def test_identical_files(self):
        from_path = self.write('one.json', '{"a": 1}\n')
        to_path = self.write('two.json', '{"a": 1}\n')
        status, _ = self.run_graphtage(from_path, to_path)
        self.assertEqual(EXIT_SUCCESS, status)

    def test_undetectable_type_is_an_error(self):
        """A failure to produce a diff is reported separately from having found differences."""
        from_path = self.write('one', '{"a": 1}\n')
        to_path = self.write('two', '{"a": 2}\n')
        status, _ = self.run_graphtage(from_path, to_path)
        self.assertEqual(EXIT_ERROR, status)

    def test_parse_failure_is_an_error(self):
        from_path = self.write('one.json', '{"a": 1}\n')
        to_path = self.write('two.json', '{this is not json\n')
        status, _ = self.run_graphtage(from_path, to_path)
        self.assertEqual(EXIT_ERROR, status)

    def test_json5_parse_failure_is_an_error(self):
        """A malformed JSON5 file used to raise TypeError from the error message's own format string."""
        from_path = self.write('one.json5', '{a: 1}\n')
        to_path = self.write('two.json5', '{a: 1,\n')
        status, _ = self.run_graphtage(from_path, to_path)
        self.assertEqual(EXIT_ERROR, status)

    def test_folded_stacks_are_detected_by_extension(self):
        """`mimetypes` does not know the flame graph extensions, so `register_mimetypes` has to add them."""
        for extension in ('folded', 'collapsed'):
            with self.subTest(extension=extension):
                from_path = self.write(f'one.{extension}', 'main;work 100\n')
                to_path = self.write(f'two.{extension}', 'main;work 120\n')
                status, output = self.run_graphtage(from_path, to_path)
                self.assertEqual(EXIT_DIFFERENCES_FOUND, status)
                self.assertIn('100 -> 120', output)

    def test_dumpversion_prints_a_bare_version_string(self):
        """`-dumpversion` joined over the version string, so it printed `0 . 3 . 1` instead of `0.3.1`."""
        out = io.StringIO()
        with contextlib.redirect_stdout(out), contextlib.redirect_stderr(io.StringIO()), \
                self.assertRaises(SystemExit) as exited:
            main(['graphtage', '-dumpversion'])
        self.assertEqual(EXIT_SUCCESS, exited.exception.code)
        self.assertEqual(version.VERSION_STRING, out.getvalue().strip())
