import os
import subprocess


def git_branch():
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


# Change DEV_BUILD to False when deploying to PyPI
DEV_BUILD = True


__version__ = (0, 1, 0)

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
