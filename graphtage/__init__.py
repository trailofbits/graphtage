from . import graphtage

from .graphtage import *
from .tree import *
from .edits import *

from .version import __version__, VERSION_STRING
from . import (
    ast, bounds, builder, constraints, dataclasses, edits, expressions, fibonacci, formatter, levenshtein, matching,
    object_set, pickle, printer, pydiff, search, sequences, tree, utils
)
from . import csv, json, plist, toml, xml, yaml

import inspect

# All of the classes in SUBMODULES_TO_SUBSUME should really be in the top-level `graphtage` module.
# They are separated into submodules solely for making the Python file sizes more manageable.
# So the following code loops over those submodules and reassigns all of the classes to the top-level module.
SUBMODULES_TO_SUBSUME = (graphtage, tree, edits)
for module_to_subsume in SUBMODULES_TO_SUBSUME:
    for name, obj in inspect.getmembers(module_to_subsume):
        if hasattr(obj, '__module__') and obj.__module__ == module_to_subsume.__name__:
            obj.__module__ = 'graphtage'
    del module_to_subsume

del inspect, SUBMODULES_TO_SUBSUME
