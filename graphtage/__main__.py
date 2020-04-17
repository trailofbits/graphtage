import argparse
import json
import logging
import os
import sys
import tempfile as tf

from tqdm import tqdm

from . import graphtage
from . import printer
from . import version
from .bounds import Range


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
        self.status: tqdm = status
        self.last_diff = None

    def __call__(self, range: Range):
        if not range.finite:
            return
        next_diff = range.upper_bound - range.lower_bound
        if self.last_diff is None:
            self.status.total = next_diff
        else:
            self.status.update(self.last_diff - next_diff)
        self.last_diff = next_diff


class make_status_callback:
    def __init__(self):
        self.status = None

    def __enter__(self):
        if not sys.stdout.isatty():
            return None
        self.status = tqdm(desc="Diffing", leave=False)
        return Callback(self.status.__enter__())

    def __exit__(self, *args, **kwargs):
        if self.status is not None:
            self.status.__exit__(*args, **kwargs)


def main(argv=None):
    parser = argparse.ArgumentParser(description='A diff utility for tree-like files such as JSON and XML.')
    parser.add_argument('FROM_PATH', type=str, nargs='?', default='-',
                        help='The source file to diff; pass \'-\' to read from STDIN')
    parser.add_argument('TO_PATH', type=str, nargs='?', default='-',
                        help='The file to diff against; pass \'-\' to read from STDIN')
    color_group = parser.add_mutually_exclusive_group()
    color_group.add_argument(
        '--color', '-c',
        action='store_true',
        default=None,
        help='Force the ANSI color output; this is turned on by default only if run from a TTY'
    )
    color_group.add_argument(
        '--no-color',
        action='store_true',
        default=None,
        help='Do not use ANSI color in the output'
    )
    parser.add_argument('--join-lists', '-jl', action='store_true',
                            help='Do not print a newline after each list entry')
    parser.add_argument('--join-dict-items', '-jd', action='store_true',
                        help='Do not print a newline after each key/value pair in a dictionary')
    parser.add_argument('--condensed', '-j', action='store_true', help='Equivalent to `-jl -jd`')
    parser.add_argument(
        '--no-key-edits',
        '-k',
        action='store_true',
        help='Only match dictionary entries if they share the same key. This drastically reduces computation.'
    )
    log_group = parser.add_mutually_exclusive_group()
    log_group.add_argument('--log-level', type=str, default='INFO', choices=list(
        logging.getLevelName(x)
        for x in range(1, 101)
        if not logging.getLevelName(x).startswith('Level')
    ), help='Sets the log level for Graphtage (default=INFO)')
    log_group.add_argument('--debug', action='store_true', help='Equivalent to `--log-level=DEBUG`')
    parser.add_argument('--version', '-v', action='store_true', help='Print Graphtage\'s version information to STDERR')
    parser.add_argument('-dumpversion', action='store_true',
                        help='Print Graphtage\'s raw version information to STDOUT and exit')

    if argv is None:
        argv = sys.argv

    args = parser.parse_args(argv[1:])

    if args.debug:
        numeric_log_level = logging.DEBUG
    else:
        numeric_log_level = getattr(logging, args.log_level.upper(), None)
        if not isinstance(numeric_log_level, int):
            sys.stderr.write(f'Invalid log level: {args.log_level}')
            exit(1)
    logging.basicConfig(level=numeric_log_level)

    if args.dumpversion:
        print(' '.join(map(str, version.__version__)))
        exit(0)

    if args.version:
        sys.stderr.write(f"Graphtage version {version.VERSION_STRING}\n")
        if args.FROM_PATH == '-' and args.TO_PATH == '-':
            exit(0)

    if args.no_color:
        ansi_color = False
    elif args.color:
        ansi_color = True
    else:
        ansi_color = None

    with PathOrStdin(args.FROM_PATH) as from_path:
        with open(from_path, 'rb') as from_file:
            from_json = json.load(from_file)
            with PathOrStdin(args.TO_PATH) as to_path:
                with open(to_path, 'rb') as to_file:
                    to_json = json.load(to_file)
                    with make_status_callback() as callback:
                        graphtage.build_tree(from_json, allow_key_edits=not args.no_key_edits).diff(
                            graphtage.build_tree(to_json, allow_key_edits=not args.no_key_edits)
                        ).print(printer.Printer(
                            sys.stdout,
                            ansi_color=ansi_color,
                            options={
                                'join_lists': args.condensed or args.join_lists,
                                'join_dict_items': args.condensed or args.join_dict_items
                            }
                        ))
#                            callback=callback


if __name__ == '__main__':
    main()
