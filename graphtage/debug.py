"""
Utilities to aid in debugging
"""

from functools import partial
from inspect import getmembers

DEBUG_MODE = False


if DEBUG_MODE:
    class Debuggable:
        _DEBUG_PATCHED: bool = False

        def __new__(cls, *args, **kwargs):
            instance = super().__new__(cls)
            if not instance._DEBUG_PATCHED:
                debug_all_member = None
                for name, member in getmembers(instance):
                    if not name.startswith("_debug_"):
                        continue
                    name = name[len("_debug_"):]
                    if name == "__all__":
                        debug_all_member = member
                        continue
                    elif not hasattr(instance, name):
                        continue
                    func = getattr(instance, name)
                    setattr(instance, f"_original_{name}", func)
                    setattr(instance, name, member)
                if debug_all_member is not None:
                    for name, member in getmembers(instance):
                        if name.startswith("_") or not callable(member):
                            continue

                        setattr(instance, name, partial(debug_all_member, name, member))
                instance._DEBUG_PATCHED = True
            return instance
else:
    class Debuggable:
        pass
