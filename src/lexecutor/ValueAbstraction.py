import json


def abstract_value(value):
    t = type(value)
    # built-in values and types that can be serialized to JSON
    if t is None or t in (bool, int, float):
        return json.dumps(value)
    # built-in sequence types
    elif t in (list, tuple, range, bytes, bytearray, memoryview,):
        return f"@@@{t.__name__}"
    # built-in set and dict types
    elif t in (set, frozenset, dict):
        return f"@@@{t.__name__}"
    # functions and methods
    elif callable(value):
        return f"@@@callable"
    # all other types
    else:
        return f"@@@object"


def restore_value(abstract_value):
    if abstract_value.startswith("@@@"):
        kind = abstract_value[3:]
        # built-in sequence types
        if kind == "list":
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
        # other object types
        else:
            return object()
    else:
        # built-in values and types that can be serialized to JSON
        return json.loads(abstract_value)
