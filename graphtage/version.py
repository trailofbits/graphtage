"""A module that centralizes the version information for Graphtage.

Changing the version here not only affects the version printed with the ``--version`` command line option, but it also
automatically updates the version used in ``setup.py`` and rendered in the documentation.

Attributes:
    DEV_BUILD (bool): Sets whether this build is a development build.
        This should only be set to :const:`True` to coincide with a release. It should *always* be :const:`True` before
        deploying to PyPI.

        If :const:`False`, the git branch will be included in :attr:`graphtage.version.__version__`.

    __version__ (Tuple[Union[int, str], ...]): The version of Graphtage. This tuple can contain any sequence of ints and
        strings. Typically this will be three ints: major/minor/revision number. However, it can contain additional
        ints and strings. If :attr:`graphtage.version.DEV_BUILD`, then `("git", git_branch())` will be appended to the
        version.

    VERSION_STRING (str): A rendered string containing the version of Graphtage. Each element of
        :attr:`graphtage.version.__version__` is appended to the string, delimited by a "." if the element is an ``int``
        or a "-" if the element is a string.

"""

import os
import subprocess
from typing import Optional, Tuple, Union


def git_branch() -> Optional[str]:
    """Returns the git branch for the codebase, or :const:`None` if it could not be determined.

    The git branch is determined by running

    .. code-block:: console

        $ git symbolic-ref -q HEAD

    """
    try:
        branch = subprocess.check_output(
            ['git', 'symbolic-ref', '-q', 'HEAD'],
            cwd=os.path.dirname(os.path.realpath(__file__)),
            stderr=subprocess.DEVNULL
        )
        branch = branch.decode('utf-8').strip().split('/')[-1]
        return branch
    except Exception:
        return None


DEV_BUILD = False
"""Sets whether this build is a development build.

This should only be set to :const:`False` to coincide with a release. It should *always* be :const:`False` before
deploying to PyPI.

If :const:`True`, the git branch will be included in the version string.

"""


__version__: Tuple[Union[int, str], ...] = (0, 2, 4)

if DEV_BUILD:
    branch_name = git_branch()
    if branch_name is None:
        __version__ = __version__ + ('git',)
    else:
        __version__ = __version__ + ('git', branch_name)

VERSION_STRING = ''

for element in __version__:
    if isinstance(element, int):
        if VERSION_STRING:
            VERSION_STRING += f'.{element}'
        else:
            VERSION_STRING = str(element)
    else:
        if VERSION_STRING:
            VERSION_STRING += f'-{element!s}'
        else:
            VERSION_STRING += str(element)


if __name__ == '__main__':
    print(VERSION_STRING)
