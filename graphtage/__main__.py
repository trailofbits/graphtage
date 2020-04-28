import argparse
import logging
import mimetypes
import os
import sys
import tempfile as tf

from . import graphtage
from . import printer as printermodule
from . import version
from .printer import HTMLPrinter, Printer


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


def main(argv=None):
    parser = argparse.ArgumentParser(
        description='A diff utility for tree-like files such as JSON, XML, HTML, YAML, and CSV.'
    )
    parser.add_argument('FROM_PATH', type=str, nargs='?', default='-',
                        help='the source file to diff; pass \'-\' to read from STDIN')
    parser.add_argument('TO_PATH', type=str, nargs='?', default='-',
                        help='the file to diff against; pass \'-\' to read from STDIN')
    file_type_group = parser.add_argument_group(title='input file types')
    file1_type_group = file_type_group.add_mutually_exclusive_group()
    file1_type_group.add_argument(
        '--from-mime',
        type=str,
        default=None,
        help='explicitly specify the MIME type of the first file',
        choices=graphtage.FILETYPES_BY_MIME.keys()
    )
    file2_type_group = file_type_group.add_mutually_exclusive_group()
    file2_type_group.add_argument(
        '--to-mime',
        type=str,
        default=None,
        help='explicitly specify the MIME type of the second file',
        choices=graphtage.FILETYPES_BY_MIME.keys()
    )
    for typename, filetype in sorted(graphtage.FILETYPES_BY_TYPENAME.items()):
        mime = filetype.default_mimetype
        file1_type_group.add_argument(
            f'--from-{typename}',
            action='store_const',
            const=mime,
            default=None,
            help=f'equivalent to `--from-mime {mime}`'
        )
        file2_type_group.add_argument(
            f'--to-{typename}',
            action='store_const',
            const=mime,
            default=None,
            help=f'equivalent to `--to-mime {mime}`'
        )
    formatting = parser.add_argument_group(title='output formatting')
    formatting.add_argument('--format', '-f', choices=graphtage.FILETYPES_BY_TYPENAME.keys(), default=None,
                            help='output format for the diff (default is to use the format of FROM_PATH)')
    color_group = formatting.add_mutually_exclusive_group()
    color_group.add_argument(
        '--color', '-c',
        action='store_true',
        default=None,
        help='force ANSI color output; this is turned on by default only if run from a TTY'
    )
    color_group.add_argument(
        '--no-color',
        action='store_true',
        default=None,
        help='do not use ANSI color in the output'
    )
    formatting.add_argument('--join-lists', '-jl', action='store_true',
                            help='do not print a newline after each list entry')
    formatting.add_argument('--join-dict-items', '-jd', action='store_true',
                        help='do not print a newline after each key/value pair in a dictionary')
    formatting.add_argument('--condensed', '-j', action='store_true', help='equivalent to `-jl -jd`')
    formatting.add_argument('--html', action='store_true', help='output the diff in HTML')
    parser.add_argument(
        '--no-key-edits',
        '-k',
        action='store_true',
        help='only match dictionary entries if they share the same key. This drastically reduces computation.'
    )
    parser.add_argument(
        '--no-status',
        action='store_true',
        help='do not display progress bars and status messages'
    )
    log_section = parser.add_argument_group(title='logging')
    log_group = log_section.add_mutually_exclusive_group()
    log_group.add_argument('--log-level', type=str, default='INFO', choices=list(
        logging.getLevelName(x)
        for x in range(1, 101)
        if not logging.getLevelName(x).startswith('Level')
    ), help='sets the log level for Graphtage (default=INFO)')
    log_group.add_argument('--debug', action='store_true', help='equivalent to `--log-level=DEBUG`')
    log_group.add_argument('--quiet', action='store_true', help='equivalent to `--log-level=CRITICAL --no-status`')
    parser.add_argument('--version', '-v', action='store_true', help='print Graphtage\'s version information to STDERR')
    parser.add_argument('-dumpversion', action='store_true',
                        help='print Graphtage\'s raw version information to STDOUT and exit')

    if argv is None:
        argv = sys.argv

    args = parser.parse_args(argv[1:])

    if args.debug:
        numeric_log_level = logging.DEBUG
    elif args.quiet:
        numeric_log_level = logging.CRITICAL
    else:
        numeric_log_level = getattr(logging, args.log_level.upper(), None)
        if not isinstance(numeric_log_level, int):
            sys.stderr.write(f'Invalid log level: {args.log_level}')
            exit(1)

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

    if args.html:
        from_file = os.path.basename(args.FROM_PATH)
        to_file = os.path.basename(args.TO_PATH)

        def printer_type(*pos_args, **kwargs):
            return HTMLPrinter(title=f"Graphtage Diff of {from_file} and {to_file}", *pos_args, **kwargs)
    else:
        printer_type = Printer

    printer = printer_type(
        sys.stdout,
        ansi_color=ansi_color,
        quiet=args.no_status or args.quiet,
        options={
            'join_lists': args.condensed or args.join_lists,
            'join_dict_items': args.condensed or args.join_dict_items
        }
    )
    printermodule.DEFAULT_PRINTER = printer

    logging.basicConfig(level=numeric_log_level, stream=Printer(
        sys.stderr,
        quiet=args.no_status or args.quiet,
    ))

    mimetypes.init()
    if '.yml' not in mimetypes.types_map and '.yaml' not in mimetypes.types_map:
        mimetypes.add_type('application/x-yaml', '.yml')
        mimetypes.suffix_map['.yaml'] = '.yml'
    elif '.yml' not in mimetypes.types_map:
        mimetypes.suffix_map['.yml'] = '.yaml'
    elif '.yaml' not in mimetypes.types_map:
        mimetypes.suffix_map['.yaml'] = '.yml'

    if args.from_mime is not None:
        from_mime = args.from_mime
    else:
        for typename in graphtage.FILETYPES_BY_TYPENAME.keys():
            from_mime = getattr(args, f'from_{typename}')
            if from_mime is not None:
                break
        else:
            from_mime = None

    if args.from_mime is not None:
        to_mime = args.from_mime
    else:
        for typename in graphtage.FILETYPES_BY_TYPENAME.keys():
            to_mime = getattr(args, f'to_{typename}')
            if to_mime is not None:
                break
        else:
            to_mime = None

    with PathOrStdin(args.FROM_PATH) as from_path:
        with PathOrStdin(args.TO_PATH) as to_path:
            from_format = graphtage.get_filetype(from_path, from_mime)
            to_format = graphtage.get_filetype(to_path, to_mime)
            from_tree = from_format.build_tree_handling_errors(from_path, allow_key_edits=not args.no_key_edits)
            to_tree = to_format.build_tree_handling_errors(to_path, allow_key_edits=not args.no_key_edits)
            diff = from_tree.diff(to_tree)
            if args.format is not None:
                formatter = graphtage.FILETYPES_BY_TYPENAME[args.format].get_default_formatter()
            else:
                formatter = from_format.get_default_formatter()
            formatter.print(printer, diff)
    printer.write('\n')
    printer.close()


if __name__ == '__main__':
    main()
