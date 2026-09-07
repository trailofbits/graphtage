import contextlib
import io
import os
import shutil
import subprocess
import sys
import tempfile
import unittest

from graphtage import git
from graphtage.__main__ import EXIT_ERROR, EXIT_SUCCESS

JSON_ONE = '{"foo": [1, 2, 3, 4], "bar": "testing"}\n'
JSON_TWO = '{"foo": [2, 3, 4, 5], "bar": "testing"}\n'
YAML_ONE = 'name: demo\nreplicas: 2\n'
YAML_TWO = 'name: demo\nreplicas: 5\n'


class GitDiffTestCase(unittest.TestCase):
    """Base class providing a scratch directory and a helper for running the diff driver."""

    def setUp(self):
        self.directory = tempfile.mkdtemp()
        self.addCleanup(shutil.rmtree, self.directory, True)

    def write(self, name, contents):
        """Writes a file into the scratch directory and returns its path."""
        path = os.path.join(self.directory, name)
        with open(path, 'w') as f:
            f.write(contents)
        return path

    def run_driver(self, *argv):
        """Runs the diff driver, returning its exit status and everything it wrote to STDOUT."""
        out = io.StringIO()
        with contextlib.redirect_stdout(out), contextlib.redirect_stderr(io.StringIO()):
            status = git.main(['graphtage-git-diff', *argv])
        return status, out.getvalue()

    def git_arguments(self, path, from_path, to_path):
        """Builds the seven arguments git passes to an external diff driver.

        The hash and mode arguments are deliberately nonsense so that a test fails if the driver ever reads them.
        """
        return [path, from_path, 'not-a-hash', 'not-a-mode', to_path, 'also-not-a-hash', 'also-not-a-mode']


class TestSplitOptions(unittest.TestCase):
    def test_no_options(self):
        self.assertEqual(([], ['a.json', 'b', 'c']), git.split_options(['a.json', 'b', 'c']))

    def test_leading_options_are_forwarded(self):
        self.assertEqual(
            (['--color', '--format=yaml'], ['a.json', 'b']),
            git.split_options(['--color', '--format=yaml', 'a.json', 'b'])
        )

    def test_the_split_stops_at_the_first_positional(self):
        """Only the leading run of options is forwarded, so later dashes stay with git's arguments."""
        self.assertEqual(
            (['-c'], ['conf.json', '--not-an-option', 'new']),
            git.split_options(['-c', 'conf.json', '--not-an-option', 'new'])
        )

    def test_empty(self):
        self.assertEqual(([], []), git.split_options([]))


class TestMimeOptions(GitDiffTestCase):
    def test_inferred_from_repository_path(self):
        self.assertEqual(
            ['--from-mime=application/json', '--to-mime=application/json'],
            git.mime_options('src/conf.json', [])
        )

    def test_forwarded_options_take_precedence(self):
        """An explicit type from the user must not collide with the inferred one."""
        self.assertEqual(['--to-mime=application/json'], git.mime_options('conf.json', ['--from-json']))
        self.assertEqual([], git.mime_options('conf.json', ['--from-mime=application/json', '--to-xml']))

    def test_unsupported_type(self):
        with self.assertRaises(ValueError):
            git.mime_options('notes.txt', [])

    def test_unknown_extension(self):
        with self.assertRaises(ValueError):
            git.mime_options('some-file-without-an-extension', [])


class TestArgumentMapping(GitDiffTestCase):
    def test_second_and_fifth_arguments_are_diffed(self):
        """Git passes the old revision as its second argument and the new one as its fifth."""
        one = self.write('one.json', JSON_ONE)
        two = self.write('two.json', JSON_TWO)
        status, output = self.run_driver(*self.git_arguments('conf.json', one, two))
        self.assertEqual(EXIT_SUCCESS, status)
        self.assertIn('conf.json', output)
        self.assertIn('5', output)

    def test_type_comes_from_the_repository_path(self):
        """Git's temporary files may have no extension, so the type must come from the first argument."""
        one = self.write('extensionless_one', YAML_ONE)
        two = self.write('extensionless_two', YAML_TWO)
        status, output = self.run_driver(*self.git_arguments('deploy/app.yaml', one, two))
        self.assertEqual(EXIT_SUCCESS, status)
        self.assertIn('replicas', output)

    def test_rename_arguments_are_accepted(self):
        """Git appends the new path and a similarity score when it detects a rename."""
        one = self.write('one.json', JSON_ONE)
        two = self.write('two.json', JSON_TWO)
        arguments = [*self.git_arguments('old.json', one, two), 'new.json', 'similarity']
        status, _ = self.run_driver(*arguments)
        self.assertEqual(EXIT_SUCCESS, status)

    def test_forwarded_options_reach_graphtage(self):
        one = self.write('one.json', JSON_ONE)
        two = self.write('two.json', JSON_TWO)
        status, output = self.run_driver('--format=yaml', *self.git_arguments('conf.json', one, two))
        self.assertEqual(EXIT_SUCCESS, status)
        self.assertNotIn('{', output)

    def test_unmerged_path(self):
        """Git passes a single argument for an unmerged path, which the driver has nothing to diff."""
        status, _ = self.run_driver('conf.json')
        self.assertEqual(EXIT_SUCCESS, status)

    def test_wrong_argument_count(self):
        status, _ = self.run_driver('conf.json', 'one', 'two')
        self.assertEqual(EXIT_ERROR, status)

    def test_option_value_as_separate_argument_is_rejected(self):
        """`--format yaml` leaves eight positional arguments, which cannot be git's."""
        one = self.write('one.json', JSON_ONE)
        two = self.write('two.json', JSON_TWO)
        status, _ = self.run_driver('--format', 'yaml', *self.git_arguments('conf.json', one, two))
        self.assertEqual(EXIT_ERROR, status)


class TestExitStatus(GitDiffTestCase):
    """Git stops the entire diff when a driver exits non-zero, so differences must not be reported as failure."""

    def test_differences_are_not_an_error(self):
        one = self.write('one.json', JSON_ONE)
        two = self.write('two.json', JSON_TWO)
        status, output = self.run_driver(*self.git_arguments('conf.json', one, two))
        self.assertEqual(EXIT_SUCCESS, status)
        self.assertIn('5', output)

    def test_identical_revisions(self):
        one = self.write('one.json', JSON_ONE)
        two = self.write('two.json', JSON_ONE)
        status, _ = self.run_driver(*self.git_arguments('conf.json', one, two))
        self.assertEqual(EXIT_SUCCESS, status)

    def test_added_file(self):
        two = self.write('two.json', JSON_TWO)
        status, output = self.run_driver(*self.git_arguments('conf.json', '/dev/null', two))
        self.assertEqual(EXIT_SUCCESS, status)
        self.assertIn('added', output)

    def test_deleted_file(self):
        one = self.write('one.json', JSON_ONE)
        status, output = self.run_driver(*self.git_arguments('conf.json', one, '/dev/null'))
        self.assertEqual(EXIT_SUCCESS, status)
        self.assertIn('deleted', output)

    def test_unsupported_type_is_an_error(self):
        one = self.write('one.txt', 'hello\n')
        two = self.write('two.txt', 'goodbye\n')
        status, _ = self.run_driver(*self.git_arguments('notes.txt', one, two))
        self.assertEqual(EXIT_ERROR, status)

    def test_parse_failure_is_an_error(self):
        one = self.write('one.json', JSON_ONE)
        two = self.write('two.json', '{this is not json\n')
        status, _ = self.run_driver(*self.git_arguments('conf.json', one, two))
        self.assertEqual(EXIT_ERROR, status)


class TestGitIntegration(unittest.TestCase):
    """Runs the driver through a real `git diff` to confirm git accepts its output and exit status."""

    def setUp(self):
        if shutil.which('git') is None:
            self.skipTest('git is not installed')
        self.directory = tempfile.mkdtemp()
        self.addCleanup(shutil.rmtree, self.directory, True)
        self.git('init', '-q')
        self.git('config', 'user.email', 'graphtage@example.com')
        self.git('config', 'user.name', 'Graphtage')
        self.git('config', 'commit.gpgsign', 'false')

    def git(self, *arguments):
        return subprocess.run(
            ['git', '-C', self.directory, *arguments],
            capture_output=True, text=True
        )

    def write(self, name, contents):
        with open(os.path.join(self.directory, name), 'w') as f:
            f.write(contents)

    def test_git_accepts_the_driver(self):
        self.write('conf.json', JSON_ONE)
        self.git('add', '-A')
        commit = self.git('commit', '-q', '-m', 'initial')
        self.assertEqual(0, commit.returncode, commit.stderr)
        self.write('conf.json', JSON_TWO)
        driver = f'"{sys.executable}" -m graphtage.git'
        environment = dict(os.environ, GIT_EXTERNAL_DIFF=driver)
        result = subprocess.run(
            ['git', '-C', self.directory, '--no-pager', 'diff'],
            capture_output=True, text=True, env=environment
        )
        self.assertEqual(0, result.returncode, result.stderr)
        self.assertIn('conf.json', result.stdout)
        self.assertIn('5', result.stdout)
