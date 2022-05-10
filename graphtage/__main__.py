import argparse
import logging
import mimetypes
import os
import sys
from abc import ABCMeta, abstractmethod
from typing import Optional

from .edits import Edit
from . import expressions
from . import graphtage
from . import printer as printermodule
from . import version
from .printer import HTMLPrinter, Printer
from .utils import Tempfile


log = logging.getLogger('graphtage')


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


class ConditionalMatcher(metaclass=ABCMeta):
    def __init__(self, condition: expressions.Expression):
        self.condition: expressions.Expression = condition

    @abstractmethod
    def __call__(self, from_node: graphtage.TreeNode, to_node: graphtage.TreeNode) -> Optional[Edit]:
        raise NotImplementedError()

    @classmethod
    def apply(cls, node: graphtage.TreeNode, condition: expressions.Expression):
        if node.edit_modifiers is None:
            node.edit_modifiers = []
        node.edit_modifiers.append(cls(condition))


class MatchIf(ConditionalMatcher):
    def __call__(self, from_node: graphtage.TreeNode, to_node: graphtage.TreeNode) -> Optional[Edit]:
        try:
            if self.condition.eval(locals={'from': from_node, 'to': to_node}):
                return None
        except Exception as e:
            log.debug(f"{e!s} while evaluating --match-if for nodes {from_node} and {to_node}")
        return graphtage.Replace(from_node, to_node)


class MatchUnless(ConditionalMatcher):
    def __call__(self, from_node: graphtage.TreeNode, to_node: graphtage.TreeNode) -> Optional[Edit]:
        try:
            if self.condition.eval(locals={'from': from_node.to_obj(), 'to': to_node.to_obj()}):
                return graphtage.Replace(from_node, to_node)
        except Exception as e:
            log.debug(f"{e!s} while evaluating --match-unless for nodes {from_node} and {to_node}")
        return None


def main(argv=None) -> int:
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
    parser.add_argument('--match-if', '-m', type=str, default=None, help='only attempt to match two dictionaries if the provided expression is satisfied. For example, `--match-if "from[\'foo\'] == to[\'bar\']"` will mean that only a dictionary which has a "foo" key that has the same value as the other dictionary\'s "bar" key will be attempted to be paired')
    parser.add_argument('--match-unless', '-u', type=str, default=None, help='similar to `--match-if`, but only attempt a match if the provided expression evaluates to `False`')
    parser.add_argument('--only-edits', '-e', action='store_true', help='only print the edits rather than a full diff')
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
    key_match_strategy = parser.add_mutually_exclusive_group()
    key_match_strategy.add_argument("--dict-strategy", "-ds", choices=("auto", "match", "none"),
                                    help="sets the strategy for matching dictionary key/value pairs: `auto` (the "
                                         "default) will automatically match two key/value pairs if they share the "
                                         "same key, but consider key edits for all non-identical keys; `match` will "
                                         "attempt to consider all possible key edits (the most computationally "
                                         "expensive); and `none` will not consider any edits on dictionary keys (the "
                                         "least computationally expensive)")
    key_match_strategy.add_argument(
        '--no-key-edits',
        '-k',
        action='store_true',
        help='only match dictionary entries if they share the same key, drastically reducing computation; this is '
             'equivalent to `--dict-strategy none`'
    )
    list_edit_group = parser.add_mutually_exclusive_group()
    list_edit_group.add_argument(
        '--no-list-edits',
        '-l',
        action='store_true',
        help='do not consider removal and insertion when comparing lists'
    )
    list_edit_group.add_argument(
        '--no-list-edits-when-same-length',
        '-ll',
        action='store_true',
        help='do not consider removal and insertion when comparing lists that are the same length'
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
    if '.json5' not in mimetypes.types_map:
        mimetypes.add_type('application/json5', '.json5')
    if '.plist' not in mimetypes.types_map:
        mimetypes.add_type('application/x-plist', '.plist')

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

    if args.match_if:
        match_if = expressions.parse(args.match_if)
    else:
        match_if = None
    if args.match_unless:
        match_unless = expressions.parse(args.match_unless)
    else:
        match_unless = None

    if args.dict_strategy == "none":
        allow_key_edits = False
        auto_match_keys = False
    elif args.dict_strategy == "auto":
        allow_key_edits = True
        auto_match_keys = True
    elif args.dict_strategy == "match":
        allow_key_edits = True
        auto_match_keys = False
    else:
        allow_key_edits = not args.no_key_edits
        auto_match_keys = allow_key_edits

    options = graphtage.BuildOptions(
        allow_key_edits=allow_key_edits,
        auto_match_keys=auto_match_keys,
        allow_list_edits=not args.no_list_edits,
        allow_list_edits_when_same_length=not args.no_list_edits_when_same_length
    )

    try:
        with printer:
            with PathOrStdin(args.FROM_PATH) as from_path:
                with PathOrStdin(args.TO_PATH) as to_path:
                    from_format = graphtage.get_filetype(from_path, from_mime)
                    to_format = graphtage.get_filetype(to_path, to_mime)
                    from_tree = from_format.build_tree_handling_errors(from_path, options)
                    if isinstance(from_tree, str):
                        sys.stderr.write(from_tree)
                        sys.stderr.write('\n\n')
                        sys.exit(1)
                    to_tree = to_format.build_tree_handling_errors(to_path, options)
                    if isinstance(to_tree, str):
                        sys.stderr.write(to_tree)
                        sys.stderr.write('\n\n')
                        sys.exit(1)
                    if match_if is not None or match_unless is not None:
                        for node in from_tree.dfs():
                            if match_if is not None:
                                MatchIf.apply(node, match_if)
                            if match_unless is not None:
                                MatchUnless.apply(node, match_unless)
                    had_edits = False
                    if args.only_edits:
                        for edit in from_tree.get_all_edits(to_tree):
                            printer.write(str(edit))
                            printer.newline()
                            had_edits = had_edits or edit.has_non_zero_cost()
                    else:
                        diff = from_tree.diff(to_tree)
                        if args.format is not None:
                            formatter = graphtage.FILETYPES_BY_TYPENAME[args.format].get_default_formatter()
                        else:
                            formatter = from_format.get_default_formatter()
                        formatter.print(printer, diff)
                        had_edits = any(any(e.has_non_zero_cost() for e in n.edit_list) for n in diff.dfs())
            printer.write('\n')
    except KeyboardInterrupt:
        sys.exit(1)
    finally:
        printer.close()
    if had_edits:
        return 1
    else:
        return 0


if __name__ == '__main__':
    sys.exit(main())
