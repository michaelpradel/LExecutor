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
    # common primitive values
    if abstract_value == "None":
        return None
    elif abstract_value == "True":
        return True
    elif abstract_value == "False":
        return False
    # built-in numeric types
    elif abstract_value == "int_neg":
        return -1
    elif abstract_value == "int_zero":
        return 0
    elif abstract_value == "int_pos":
        return 1
    elif abstract_value == "float_neg":
        return -1.0
    elif abstract_value == "float_zero":
        return 0.0
    elif abstract_value == "float_pos":
        return 1.0
    # built-in sequence types
    elif abstract_value == "list_empty":
        return []
    elif abstract_value == "list_nonempty":
        # TODO return special object that delays evaluation of its elements (also for other collection types)
        return [object()]
    elif abstract_value == "tuple_empty":
        return ()
    elif abstract_value == "tuple_nonempty":
        return (object(),)
    # built-in set and dict types
    elif abstract_value == "set_empty":
        return set()
    elif abstract_value == "set_nonempty":
        return {object()}
    elif abstract_value == "dict_empty":
        return {}
    elif abstract_value == "dict_nonempty":
        return {"a": object()}
    # functions and methods
    elif abstract_value == "resource":
        return MyResource()
    elif abstract_value == "callable":
        return dummy_function
    # all other types
    else:
        return object()

    # TODO If we had a way to "taint" all injected values, could decide more precisely in Runtime.mode_branch about which exceptions to catch
