"""A module that centralizes the version information for Graphtage.

Changing the version here not only affects the version printed with the ``--version`` command line option, but it also
automatically updates the version used in the build system and rendered in the documentation.
"""

__version__ = "0.3.1"
VERSION_STRING = __version__

# For backwards compatibility with code that expects the tuple format
__version_tuple__ = tuple(int(x) if x.isdigit() else x for x in __version__.split('.'))


if __name__ == '__main__':
    print(VERSION_STRING)
