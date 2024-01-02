"""
Utilities to aid in debugging
"""

from inspect import getmembers

DEBUG_MODE = False


if DEBUG_MODE:
    class Debuggable:
        _DEBUG_PATCHED: bool = False

        def __new__(cls, *args, **kwargs):
            instance = super().__new__(cls)
            if not instance._DEBUG_PATCHED:
                for name, member in getmembers(instance):
                    if not name.startswith("_debug_"):
                        continue
                    name = name[len("_debug_"):]
                    if not hasattr(instance, name):
                        continue
                    func = getattr(instance, name)
                    setattr(instance, f"_original_{name}", func)
                    setattr(instance, name, member)
                instance._DEBUG_PATCHED = True
            return instance
else:
    class Debuggable:
        pass
