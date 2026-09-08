"""A ``git`` external diff driver that formats changes with Graphtage.

Git runs an external diff driver with seven positional arguments::

    path old-file old-hex old-mode new-file new-hex new-mode

``old-file`` and ``new-file`` are the two revisions to compare. Git usually passes at least one of them as a
temporary copy whose name does not necessarily carry the original file extension, so this driver resolves the file
type from ``path``, which is the file's location in the repository. Git appends two more arguments when it detects
a rename or a copy, and passes a single argument for an unmerged path; both forms are handled.

Git stops the whole diff when a driver exits with a non-zero status, so this driver reports a successful diff as
:const:`graphtage.__main__.EXIT_SUCCESS` even when the two revisions differ. A status of
:const:`graphtage.__main__.EXIT_ERROR` is reserved for failures that prevent Graphtage from producing a diff at
all, such as an unsupported file type or a file that does not parse.

See the "Git Integration" section of the Graphtage README for the ``git`` configuration this driver expects.
"""

import os
import sys
from collections.abc import Sequence

from . import graphtage
from .__main__ import EXIT_DIFFERENCES_FOUND, EXIT_ERROR, EXIT_SUCCESS, register_mimetypes
from .__main__ import main as graphtage_main

UNMERGED_ARGUMENT_COUNT = 1
"""The number of arguments git passes for an unmerged path."""

DIFF_ARGUMENT_COUNTS = (7, 9)
"""The number of arguments git passes for a changed path, without and with rename or copy detection."""

ABSENT_FILE_NAMES = frozenset(('/dev/null', os.devnull))
"""The names git uses in place of a revision that does not exist, such as when a file is added or deleted."""

USAGE = """usage: graphtage-git-diff [GRAPHTAGE_OPTION...] PATH OLD_FILE OLD_HEX OLD_MODE NEW_FILE NEW_HEX NEW_MODE

This command implements git's external diff interface, so git supplies the positional arguments. Any Graphtage
options must come first and must spell values with an equals sign, as in `--format=yaml`. See the "Git Integration"
section of the Graphtage README.
"""


def split_options(argv: Sequence[str]) -> tuple[list[str], list[str]]:
    """Splits the leading Graphtage options off of the arguments that git supplies.

    Git appends its own positional arguments after whatever command the user configured, so every argument up to
    the first one that does not start with ``-`` belongs to the user. Two consequences follow: an option that takes
    a value must spell it with an equals sign, because a value passed as a separate argument is indistinguishable
    from the first argument supplied by git, and a repository path that starts with ``-`` is read as an option.

    Args:
        argv: The command line arguments, excluding the name of the program.

    Returns:
        Tuple[List[str], List[str]]: The options to forward to Graphtage, followed by git's arguments.
    """
    index = 0
    while index < len(argv) and argv[index].startswith('-'):
        index += 1
    return list(argv[:index]), list(argv[index:])


def mime_options(path: str, forwarded: Sequence[str]) -> list[str]:
    """Builds the Graphtage MIME type options implied by a path in the repository.

    Args:
        path: The path of the file in the repository, which git passes as its first argument.
        forwarded: The Graphtage options that the user configured, which take precedence over the inferred type.

    Returns:
        List[str]: The ``--from-mime`` and ``--to-mime`` options that are not already covered by ``forwarded``.

    Raises:
        ValueError: If the file type of ``path`` is unknown or is not supported by Graphtage.
    """
    prefixes = [
        prefix for prefix in ('--from-', '--to-')
        if not any(option.startswith(prefix) for option in forwarded)
    ]
    if not prefixes:
        return []
    mime_type = graphtage.get_filetype(path).default_mimetype
    return [f'{prefix}mime={mime_type}' for prefix in prefixes]


def absent_revision(from_path: str, to_path: str) -> str | None:
    """Describes a change that leaves Graphtage with only one revision to work from.

    Args:
        from_path: The old revision, which git passes as its second argument.
        to_path: The new revision, which git passes as its fifth argument.

    Returns:
        str | None: ``'added'`` or ``'deleted'`` if one of the revisions is missing, and :const:`None` if both
        of them exist.
    """
    if from_path in ABSENT_FILE_NAMES:
        return 'added'
    if to_path in ABSENT_FILE_NAMES:
        return 'deleted'
    return None


def main(argv: Sequence[str] | None = None) -> int:
    """Runs Graphtage over the two revisions that git supplies.

    Args:
        argv: The command line arguments, including the name of the program. Defaults to :data:`sys.argv`.

    Returns:
        int: :const:`graphtage.__main__.EXIT_SUCCESS` if the diff was produced, whether or not the revisions differ,
        and :const:`graphtage.__main__.EXIT_ERROR` if Graphtage could not produce one.
    """
    if argv is None:
        argv = sys.argv
    forwarded, git_args = split_options(argv[1:])
    if len(git_args) == UNMERGED_ARGUMENT_COUNT:
        sys.stderr.write(f"graphtage: skipping unmerged path {git_args[0]}\n")
        return EXIT_SUCCESS
    if len(git_args) not in DIFF_ARGUMENT_COUNTS:
        sys.stderr.write(USAGE)
        return EXIT_ERROR
    path, from_path, to_path = git_args[0], git_args[1], git_args[4]
    sys.stdout.write(f"{path}\n")
    change = absent_revision(from_path, to_path)
    if change is not None:
        sys.stdout.write(f"({change}; Graphtage compares two revisions and needs both of them)\n")
        return EXIT_SUCCESS
    register_mimetypes()
    try:
        inferred = mime_options(path, forwarded)
    except ValueError as e:
        sys.stderr.write(f"Error: {e!s}\n")
        return EXIT_ERROR
    sys.stdout.flush()
    status = graphtage_main(['graphtage', '--no-status', *forwarded, *inferred, from_path, to_path])
    if status == EXIT_DIFFERENCES_FOUND:
        return EXIT_SUCCESS
    return status


if __name__ == '__main__':
    sys.exit(main())
