import inspect

from . import graphtage

for name, obj in inspect.getmembers(graphtage, inspect.isclass):
    if obj.__module__ == 'graphtage.graphtage':
        obj.__module__ = 'graphtage'

from .graphtage import *

del inspect

from .version import __version__, VERSION_STRING
from . import bounds, edits, expressions, fibonacci, formatter, levenshtein, matching, printer, \
                                                               search, sequences, tree, utils
from . import csv, json, xml, yaml
