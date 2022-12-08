import json


def abstract_value(value):
    t = type(value)
    # common values primitive values
    if value is None:
        return "@None"
    elif value is True:
        return "@True"
    elif value is False:
        return "@False"
    # built-in numeric or sequence types
    elif t in (int, float, list, tuple, range, bytes, bytearray, memoryview,):
        return f"@{t.__name__}"
    # built-in set and dict types
    elif t in (set, frozenset, dict):
        return f"@{t.__name__}"
    # functions and methods
    elif callable(value):
        if hasattr(value, "__enter__") and hasattr(value, "__exit__"):
            return f"@resource"
        else:
            return f"@callable"
    # all other types
    else:
        return f"@object"


class MyResource(object):
    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, trace):
        return True


def dummy_function(*a, **b):
    return ()


def restore_value(abstract_value):
    # TODO If we had a way to "taint" all injected values, could decide more precisely in Runtime.mode_branch about which exceptions to catch

    # common primitives
    if abstract_value == "@None":
        return None
    elif abstract_value == "@True":
        return True
    elif abstract_value == "@False":
        return False
    # numeric types
    elif abstract_value == "int":
        return 0
    elif abstract_value == "float":
        return 0.0
    # built-in sequence types
    elif abstract_value == "list":
        return []
    elif abstract_value == "tuple":
        return ()
    elif abstract_value == "range":
        return range(0)
    elif abstract_value == "bytes":
        return bytearray()
    elif abstract_value == "memoryview":
        return memoryview(bytearray())
    # built-in set and dict types
    elif abstract_value == "set":
        return set()
    elif abstract_value == "frozenset":
        return frozenset()
    elif abstract_value == "dict":
        return {}
    # functions and methods
    elif abstract_value == "callable":
        return dummy_function
    # resources (to be used in 'with' statements)
    elif abstract_value == "resource":
        return MyResource()
    # other object types
    else:
        return object()
