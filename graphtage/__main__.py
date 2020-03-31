import argparse
import json
import os
import sys
import tempfile as tf
from typing import Optional

from tqdm import tqdm

from . import graphtage
from .search import Range
from . import version


class Tempfile:
    def __init__(self, contents, prefix=None, suffix=None):
        self._temp = None
        self._data = contents
        self._prefix = prefix
        self._suffix = suffix

    def __enter__(self):
        self._temp = tf.NamedTemporaryFile(prefix=self._prefix, suffix=self._suffix, delete=False)
        self._temp.write(self._data)
        self._temp.flush()
        self._temp.close()
        return self._temp.name

    def __exit__(self, type, value, traceback):
        if self._temp is not None:
            os.unlink(self._temp.name)
            self._temp = None


class PathOrStdin:
    def __init__(self, path):
        self._path = path
        if self._path == '-':
            self._tempfile = Tempfile(sys.stdin.buffer.read())
        else:
            self._tempfile = None

    def __enter__(self):
        if self._tempfile is None:
            return self._path
        else:
            return self._tempfile.__enter__()

    def __exit__(self, *args, **kwargs):
        if self._tempfile is not None:
            return self._tempfile.__exit__(*args, **kwargs)


class Callback:
    def __init__(self, status: tqdm):
        self.total_set = False
        self.status: tqdm = status

    def __call__(self, range: Range):
        if not range.finite:
            return
        if not self.total_set:
            self.status.total = range.upper_bound - range.lower_bound
            self.total_set = True
        self.status.update(self.status.total - (range.upper_bound - range.lower_bound))


class make_status_callback:
    def __init__(self):
        self.status = None

    def __enter__(self):
        if not sys.stdout.isatty():
            return None
        self.status = tqdm(desc="Diffing", leave=False)
        return Callback(self.status.__enter__())

    def __exit__(self, *args, **kwargs):
        self.status.__exit__(*args, **kwargs)


def main(argv=None):
    parser = argparse.ArgumentParser(description='A diff utility for tree-like files such as JSON and XML.')
    parser.add_argument('FROM_PATH', type=str, nargs='?', default='-',
                        help='The source file to diff; pass \'-\' to read from STDIN')
    parser.add_argument('TO_PATH', type=str, nargs='?', default='-',
                        help='The file to diff against; pass \'-\' to read from STDIN')
    parser.add_argument('--version', '-v', action='store_true', help='Print Graphtage\'s version information to STDERR')
    parser.add_argument('-dumpversion', action='store_true',
                        help='Print Graphtage\'s raw version information to STDOUT and exit')

    if argv is None:
        argv = sys.argv

    args = parser.parse_args(argv[1:])

    if args.dumpversion:
        print(' '.join(map(str, version.__version__)))
        exit(0)

    if args.version:
        sys.stderr.write(f"Graphtage version {version.VERSION_STRING}\n")
        if args.FROM_PATH == '-' and args.TO_PATH == '-':
            exit(0)

    with PathOrStdin(args.FROM_PATH) as from_path:
        with open(from_path, 'rb') as from_file:
            from_json = json.load(from_file)
            with PathOrStdin(args.TO_PATH) as to_path:
                with open(to_path, 'rb') as to_file:
                    to_json = json.load(to_file)
                    with make_status_callback() as callback:
                        diff = graphtage.diff(
                            graphtage.build_tree(from_json),
                            graphtage.build_tree(to_json),
                            callback=callback
                        )

                    diff.print(graphtage.Printer(sys.stdout))


if __name__ == '__main__':
    main()
