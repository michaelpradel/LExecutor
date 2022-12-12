def abstract_value(value):
    t = type(value)
    # common primitive values
    if value is None:
        return "@None"
    elif value is True:
        return "@True"
    elif value is False:
        return "@False"
    # built-in numeric types
    elif t is int:
        if value < 0:
            return "@int_neg"
        elif value == 0:
            return "@int_zero"
        else:
            return "@int_pos"
    elif t is float:
        if value < 0:
            return "@float_neg"
        elif value == 0:
            return "@float_zero"
        else:
            return "@float_pos"
    # built-in sequence types
    elif t is list:
        if len(value) == 0:
            return "@list_empty"
        else:
            return "@list_nonempty"
    elif t is tuple:
        if len(value) == 0:
            return "@tuple_empty"
        else:
            return "@tuple_nonempty"
    # built-in set and dict types
    elif t is set:
        if len(value) == 0:
            return "@set_empty"
        else:
            return "@set_nonempty"
    elif t is dict:
        if len(value) == 0:
            return "@dict_empty"
        else:
            return "@dict_nonempty"
    # functions and methods
    elif callable(value):
        if hasattr(value, "__enter__") and hasattr(value, "__exit__"):
            return "@resource"
        else:
            return "@callable"
    # all other types
    else:
        return "@object"


class MyResource(object):
    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, trace):
        return True


def dummy_function(*a, **b):
    return ()


def restore_value(abstract_value):
    # TODO adapt to the revised value abstraction

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
