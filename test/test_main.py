import contextlib
import io
import os
import shutil
import tempfile
import unittest

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
