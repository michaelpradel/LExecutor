import json


def abstract_value(value):
    t = type(value)
    # common values that can be serialized to JSON
    if value is None or t is bool:
        return json.dumps(value)
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


def restore_value(abstract_value):
    if abstract_value.startswith("@"):
        kind = abstract_value[3:]
        # numeric types
        if kind == "int":
            return 0
        elif kind == "float":
            return 0.0
        # built-in sequence types
        elif kind == "list":
            return []
        elif kind in "tuple":
            return ()
        elif kind == "range":
            return range(0)
        elif kind == "bytes":
            return bytearray()
        elif kind == "memoryview":
            return memoryview(bytearray())
        # built-in set and dict types
        elif kind == "set":
            return set()
        elif kind == "frozenset":
            return frozenset()
        elif kind == "dict":
            return {}
        # functions and methods
        elif kind == "callable":
            return lambda *a, **b: ()
        # resources (to be used in 'with' statements)
        elif kind == "resource":
            return MyResource()
        # other object types
        else:
            return object()
    else:
        # built-in values and types that can be serialized to JSON
        return json.loads(abstract_value)
